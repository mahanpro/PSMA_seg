"""
Usage:
python psma_cleaner.py --in_jsonl cache/text/raw_reports.jsonl --out_jsonl cache/text/clean_reports.jsonl
"""

# -*- coding: utf-8 -*-

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Any
from typing import Match


# ---- Normalization dictionaries/patterns ----

TYPO_MAP = {
    "heterogenous": "heterogeneous",
    "likey": "likely",
    "most likey": "most likely",
    "non specific": "non-specific",
    "nonmalignant": "non-malignant",
    "non malignant": "non-malignant",
    "desciption": "description",
    "proistate": "prostate",
    "kifney": "kidney",
    "ischiopuboc": "ischiopubic",
    "ischopubic": "ischiopubic",
    "acetabulom": "acetabulum",
    "blladder": "bladder",
    "hypoaatenuating": "hypoattenuating",
    "prominanlty": "prominently",
    "prominant": "prominent",
    "prominately": "prominently",
    "schomrl": "schmorl",
    "endlates": "endplates",
}

PSMA_VARIANTS = [
    (
        re.compile(r"\bPSMA\s*\-\s*positive\b", re.I),
        "PSMA-positive",
    ),  # PSMA - positive --> PSMA-positive
    (
        re.compile(r"\bPSMA\s*positive\b", re.I),
        "PSMA-positive",
    ),  # PSMA  positive  --> PSMA-positive
    (
        re.compile(r"\bPSMA\s*\-\s*negative\b", re.I),
        "PSMA-negative",
    ),  # PSMA - negative --> PSMA-negative
    (
        re.compile(r"\bPSMA\s*negative\b", re.I),
        "PSMA-negative",
    ),  # PSMA  negative  --> PSMA-negative
]
SUV_VARIANTS = [
    (re.compile(r"\bSUV\s*max\b", re.I), "SUVmax"),  # SUV max --> SUVmax
    (re.compile(r"\bSUV\s*Max\b", re.I), "SUVmax"),  # SUV Max --> SUVmax
    (re.compile(r"\bSUV\s*MAX\b", re.I), "SUVmax"),  # SUV MAX --> SUVmax
]
HEADER_VARIANTS = [
    (
        re.compile(r"(?mi)\bProstatic\s*fossa\s*:\s*"),
        "Prostatic fossa: ",
    ),  # PROSTATIC FOSSA: --> Prostatic fossa:
    (
        re.compile(r"(?mi)\bLymph\s*nodes\s*:\s*"),
        "Lymph nodes: ",
    ),  # Lymph nodes : --> Lymph nodes:
    (re.compile(r"(?mi)\bSkeleton\s*:\s*"), "Skeleton: "),  # Skeleton : --> Skeleton:
    (re.compile(r"(?mi)\bViscera\s*:\s*"), "Viscera: "),  # Viscera : --> Viscera:
]

# Measurements: "a x b mm/cm"
MEAS = re.compile(r"(?i)(\d+(?:\.\d+)?)\s*[x×]\s*(\d+(?:\.\d+)?)\s*(mm|cm)\b")
SUV_EXTRACT = re.compile(r"(?i)\bSUVmax\s*=\s*([<>]?\d+(?:\.\d+)?)")


def normalize_text(s: str) -> str:
    # Preserve line breaks; only collapse spaces/tabs
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u00d7", "x")
    s = re.sub(r"[ \t]+", " ", s)

    # Whole-word typo fixes (do not remove newlines)
    def _fix_word(m: Match[str]) -> str:
        w = m.group(0)
        return TYPO_MAP.get(w.lower(), w)

    s = re.sub(r"[A-Za-z][A-Za-z\-]*", _fix_word, s)

    for pat, rep in PSMA_VARIANTS:
        s = pat.sub(rep, s)
    for pat, rep in SUV_VARIANTS:
        s = pat.sub(rep, s)
    for pat, rep in HEADER_VARIANTS:
        s = pat.sub(rep, s)

    # Tidy spaces around punctuation without touching \n
    s = re.sub(r"\s*:\s*", ": ", s)
    s = re.sub(r"\(\s+", "(", s)
    s = re.sub(r"\s+\)", ")", s)
    return s.strip()


DESC_SPLIT = re.compile(r"(?mi)^\s*Description\s*:\s*")


def split_reports_from_blob(blob: str) -> List[str]:
    parts = DESC_SPLIT.split(blob)
    out = []
    for p in parts[1:]:
        rpt = "Description: " + p.strip()
        if rpt:
            out.append(rpt)
    return out


SEC_PATTERNS = {
    "Prostatic fossa": re.compile(
        r"(?is)Prostatic fossa:\s*(.*?)(?=Lymph nodes:|Skeleton:|Viscera:|\Z)"
    ),
    "Lymph nodes": re.compile(
        r"(?is)Lymph nodes:\s*(.*?)(?=Prostatic fossa:|Skeleton:|Viscera:|\Z)"
    ),
    "Skeleton": re.compile(
        r"(?is)Skeleton:\s*(.*?)(?=Prostatic fossa:|Lymph nodes:|Viscera:|\Z)"
    ),
    "Viscera": re.compile(
        r"(?is)Viscera:\s*(.*?)(?=Prostatic fossa:|Lymph nodes:|Skeleton:|\Z)"
    ),
}


def parse_sections(text: str) -> Dict[str, str]:
    out = {}
    for name, pat in SEC_PATTERNS.items():
        m = pat.search(text)
        if m:
            out[name] = m.group(1).strip()
    return out


def extract_measurements(text: str) -> List[Dict[str, Any]]:
    vals = []
    for a, b, unit in MEAS.findall(text):
        try:
            vals.append({"a": float(a), "b": float(b), "unit": unit.lower()})
        except Exception:
            pass
    return vals


def extract_suv_values(text: str) -> List[float]:
    vals: List[float] = []
    for v in SUV_EXTRACT.findall(text):
        try:
            vals.append(float(v.replace("<", "").replace(">", "")))
        except Exception:
            pass
    return vals


def find_flags(clean: str) -> List[str]:
    flags: List[str] = []
    if re.search(r"\d+[A-Za-z]\d*\.\d+", clean):
        flags.append("odd_numeric_token")
    if re.search(r"[^\x00-\x7F]", clean):
        flags.append("non_ascii")
    if clean.count("(") != clean.count(")"):
        flags.append("paren_mismatch")
    for sec in ("Prostatic fossa", "Lymph nodes", "Skeleton", "Viscera"):
        if sec not in clean:
            flags.append(f"missing_section:{sec}")
    if "Benign or malignant?" in clean or "Benign or malignant" in clean:
        flags.append("explicit_question_benign_or_malignant")
    return flags


def main():
    ap = argparse.ArgumentParser(description="Clean PSMA reports")
    ap.add_argument(
        "--in_jsonl", type=str, help="raw_reports.jsonl ({id, description})"
    )
    ap.add_argument(
        "--out_jsonl", type=str, required=True, help="Output clean_reports.jsonl"
    )
    args = ap.parse_args()

    records: List[Dict[str, Any]] = []

    for line in Path(args.in_jsonl).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        rid = obj["ID"]
        description = obj["Description"]
        records.append({"ID": rid, "Description": description})

    outp = Path(args.out_jsonl)
    outp.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with outp.open("w", encoding="utf-8") as f:
        for rec in records:
            rid = rec["ID"]
            description = unicodedata.normalize("NFKC", rec["Description"])
            clean = normalize_text(description)
            secs = parse_sections(clean)
            suv = extract_suv_values(clean)
            meas = extract_measurements(clean)
            flags = find_flags(clean)
            row = {
                "ID": rid,
                "clean": clean,
                "sections": secs,
                "suvmax": suv,
                "measurements": meas,
                "flags": flags,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1

    print(f"Wrote {n} records → {outp}")


if __name__ == "__main__":
    main()
