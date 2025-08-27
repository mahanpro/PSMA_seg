"""
Usage:
python psma_cleaner.py --in_jsonl cache/text/raw_reports.jsonl --out_jsonl cache/text/clean_reports.jsonl
"""

# coding: utf-8

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Any
from typing import Match
import hashlib


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
    "lesin": "lesion",
    "leion": "lesion",
    "bronchestatic": "bronchiectatic",
    "midly": "mildly",
    "extenstion": "extension",
    "lunge": "lung",
    "negatic": "negative",
    "tisse": "tissue",
    "peripheraly": "peripherally",
    "paraaortic": "para-aortic",
    "prominanrtly": "prominently",
    "prominnatly": "prominently",
    "prominatlyon": "prominently on",
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
MEAS = re.compile(
    r"(?ix)"
    r"(\d+(?:\.\d+)?)\s*[x×]\s*"
    r"(?:0?\.(\d+)|(\d+(?:\.\d+)?))\s*"  # capture ".4" or "0.4" or "1.2"
    r"(mm|cm)\b"
)
SUVMAX_CLAUSE = re.compile(
    r"(?i)\bSUVmax\b(?:(?:\.(?=\d))|[^.;)\n])*"  # grab the clause after 'SUVmax' until sentence-ish boundary
)

NUM_F = re.compile(r"[<>]?\d+(?:\.\d+)?")  # numbers incl. inequalities


def extract_suv_values_from_section(
    sections: Dict[str, str], section_name: str
) -> List[float]:
    """
    Extract all SUVmax numeric values from a given section.
    Returns [] if the section is missing.
    """
    sec = sections.get(section_name, "")
    out: List[float] = []
    if not sec:
        return out
    for m in SUVMAX_CLAUSE.finditer(sec):
        clause = m.group(0)
        # --- local numeric repairs confined to the SUV clause ---
        clause = re.sub(r"(\d)\s*\.\s*(\d)", r"\1.\2", clause)  # "7 . 8" -> "7.8"
        clause = re.sub(
            r"=\s*\d+\s+(\d+\.\d+)", r"= \1", clause
        )  # "= 7  78.3" -> "= 78.3"
        # ---------------------------------------------------------
        for n in NUM_F.findall(clause):
            out.append(float(n.replace("<", "").replace(">", "")))
    return out


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

    #### new
    # 1) Fix 'PSM a' split token
    s = re.sub(r"\bPSM\s+a\b", "PSMA", s, flags=re.I)

    # 2) Ensure space before '(' when preceded by letters
    s = re.sub(r"(?<=[A-Za-z])\(", " (", s)

    # 3) Collapse double commas and extra spaces before periods/commas
    s = re.sub(r",\s*,+", ", ", s)  # ",," -> ", "
    s = re.sub(r"\s+([.,;])", r"\1", s)  # "word ." -> "word."

    # 4) Insert space before units (cm/mm)
    s = re.sub(r"(\d)(?=(?:mm|cm)\b)", r"\1 ", s, flags=re.I)

    # 5) Add leading zero to decimals in measurements contexts
    s = re.sub(r"(?i)(x\s*)\.(\d+)\s*(mm|cm)\b", r"\g<1>0.\2 \3", s)

    # 6) Normalize lone '(Max = N)' to '(SUVmax = N)' when near PSMA text
    s = re.sub(
        r"(?i)(PSMA[^()\n]{0,80})\((?:\s*)Max\s*=\s*([<>]?\d+(?:\.\d+)?)\s*\)",
        lambda m: f"{m.group(1)}(SUVmax = {m.group(2)})",
        s,
    )

    # 7) Repair split SUV numbers like 'SUVmax = 7 78.3' -> 'SUVmax = 78.3'
    s = re.sub(r"(?i)(SUV\s*max|SUVmax)\s*=\s*\d+\s+(\d+\.\d+)", r"SUVmax = \2", s)

    # 8) Very specific salvage for a letter embedded inside a number (e.g., '4D7.0')
    s = re.sub(r"(?<=\d)[A-Z](?=\d*\.\d+)", "", s)

    # 9) Normalize common 'promin...' family loosely
    s = re.sub(r"\bprominanlty\b", "prominently", s, flags=re.I)
    s = re.sub(r"\bprominant(?:ly)?\b", "prominent", s, flags=re.I)
    s = re.sub(r"\bprominately\b", "prominently", s, flags=re.I)

    # 10) Remove stray punctuation after 'SUVmax = <number>' (keep any closing bracket)
    #     e.g. "SUVmax = 10.5." or "(SUVmax = 10.5)." -> "SUVmax = 10.5" / "(SUVmax = 10.5)"
    s = re.sub(
        r"(?i)(SUV\s*max|SUVmax)\s*=\s*([<>]?\d+(?:\.\d+)?)(\s*[\)\]\}])?\s*[.,;](?!\d)",
        r"SUVmax = \2\3",
        s,
    )
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
        r"(?is)Prostatic fossa:\s*(.*?)(?=\n\s*\n|Lymph nodes:|Skeleton:|Viscera:|\Z)"
    ),
    "Lymph nodes": re.compile(
        r"(?is)Lymph nodes:\s*(.*?)(?=\n\s*\n|Prostatic fossa:|Skeleton:|Viscera:|\Z)"
    ),
    "Skeleton": re.compile(
        r"(?is)Skeleton:\s*(.*?)(?=\n\s*\n|Prostatic fossa:|Lymph nodes:|Viscera:|\Z)"
    ),
    "Viscera": re.compile(
        r"(?is)Viscera:\s*(.*?)(?=\n\s*\n|Prostatic fossa:|Lymph nodes:|Skeleton:|\Z)"
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
    vals: List[Dict[str, Any]] = []
    # MEAS now returns (a, b_decimals_only, b_full, unit)
    for a, b_dec, b_full, unit in MEAS.findall(text):
        # if ".4" matched, b_dec holds "4"; otherwise b_full holds something like "0.4" or "1.2"
        b_str = f"0.{b_dec}" if b_dec else b_full
        try:
            vals.append({"a": float(a), "b": float(b_str), "unit": unit.lower()})
        except Exception:
            pass
    return vals


def extract_suv_values(text: str) -> List[float]:
    """
    Global SUVmax values across the whole text.
    """
    vals: List[float] = []
    for m in SUVMAX_CLAUSE.finditer(text):
        clause = m.group(0)
        # --- local numeric repairs confined to the SUV clause ---
        clause = re.sub(r"(\d)\s*\.\s*(\d)", r"\1.\2", clause)
        clause = re.sub(r"=\s*\d+\s+(\d+\.\d+)", r"= \1", clause)
        # ---------------------------------------------------------
        for n in NUM_F.findall(clause):
            vals.append(float(n.replace("<", "").replace(">", "")))
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
    ##### new
    if re.search(r"(?i)\((?:\s*)Max\s*=\s*[<>]?\d", clean):
        flags.append("dangling_max_equals")

    if re.search(r"(?i)(SUV\s*max|SUVmax)\s*=\s*\d+\s+\d+\.\d+", clean):
        flags.append("suv_split_number")

    if re.search(r"(?i)x\s*\.\d+\s*(mm|cm)\b", clean):
        flags.append("leading_decimal_no_zero")

    if re.search(r"(?i)\bstatus post prostatectomy\b", clean) and re.search(
        r"\bbilateral prostate lobes?\b", clean
    ):
        flags.append("prostatectomy_lobes_contradiction")

    if (
        re.search(r"[^\s\w.,:;()<>=%/-]", clean)
        and "?" in clean
        and not re.search(r"\b(PSMA|SUVmax|cm|mm)\b", clean)
    ):
        flags.append("noise_line_or_gibberish")

    if ",," in clean:
        flags.append("double_comma")
    return flags


def _hash_for_dedupe(text: str) -> str:
    base = re.sub(r"\s+", " ", text.lower()).strip()
    return hashlib.sha1(base.encode("utf-8")).hexdigest()


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
    seen_hashes = set()
    with outp.open("w", encoding="utf-8") as f:
        for rec in records:
            rid = rec["ID"]
            description = unicodedata.normalize("NFKC", rec["Description"])
            clean = normalize_text(description)
            secs = parse_sections(clean)
            suv_all = extract_suv_values(clean)
            suv_pf = extract_suv_values_from_section(secs, "Prostatic fossa")
            suv_ln = extract_suv_values_from_section(secs, "Lymph nodes")
            suv_skeleton = extract_suv_values_from_section(secs, "Skeleton")
            suv_viscera = extract_suv_values_from_section(secs, "Viscera")
            meas = extract_measurements(clean)
            flags = find_flags(clean)

            row = {
                "ID": rid,
                "clean": clean,
                "sections": secs,
                "suvmax_all": suv_all,
                "suvmax_prostatic_fossa": suv_pf,
                "suvmax_lymph_nodes": suv_ln,
                "suvmax_skeleton": suv_skeleton,
                "suvmax_viscera": suv_viscera,
                "measurements": meas,
                "flags": flags,
            }

            # ---- duplicate detection & flagging ----
            row_hash = _hash_for_dedupe(clean)
            row["dedupe_hash"] = row_hash
            if row_hash in seen_hashes:
                row["flags"].append("duplicate_clean_text")
            else:
                seen_hashes.add(row_hash)
            # ---------------------------------------

            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1

    print(f"Wrote {n} records → {outp}")


if __name__ == "__main__":
    main()
