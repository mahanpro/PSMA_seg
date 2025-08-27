#!/usr/bin/env python3
"""
Usage:
python gpt_embed.py \
  --in_clean_jsonl cache/text/clean_reports.jsonl \
  --out_dir        cache/text \
  --extract_model  gpt-5-mini \
  --embed_model    text-embedding-3-small \
  --max_examples   3 \
  --embed_mode     both \
  --seed           13
"""

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)

# coding: utf-8
import argparse, json, os, re, time, hashlib, math, random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# pip install openai jsonschema torch numpy
from openai import OpenAI
from openai import APIError, APIConnectionError, RateLimitError, Timeout
from jsonschema import validate, ValidationError
import torch
import numpy as np

SECTIONS = ["Prostatic fossa", "Lymph nodes", "Skeleton", "Viscera"]

# ---------------- JSON Schema ----------------
FINDING_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "section": {"type": "string", "enum": SECTIONS + ["Other", "Unknown"]},
        "site": {"type": "string"},  # e.g., "right external iliac lymph node"
        "anatomy": {"type": "string"},  # optional finer-grain
        "laterality": {
            "type": ["string", "null"],
            "enum": ["left", "right", "bilateral", "both", None],
        },
        "psma_status": {"type": "string", "enum": ["positive", "negative", "n/a"]},
        "status": {"type": "string", "enum": ["present", "absent", "uncertain"]},
        "negation_cue": {"type": "string"},
        "suvmax": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "raw_value_str": {"type": "string"},  # preserve digits exactly
                    "value": {"type": "number"},
                    "comparator": {
                        "type": "string",
                        "enum": ["=", "<", ">", "<=", ">=", "upto"],
                    },
                },
                "required": ["raw_value_str", "value"],
            },
        },
        "measurements": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                    "unit": {"type": "string"},  # "mm" or "cm"
                    "mm": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 1,
                        "maxItems": 3,
                    },  # normalized to mm
                },
                "required": ["a", "b", "unit"],
            },
        },
        "evidence": {"type": "string"},
        "span": {
            "type": "array",
            "items": {"type": "integer"},
            "minItems": 2,
            "maxItems": 2,
        },
        "invades": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "section",
        "site",
        "psma_status",
        "status",
        "suvmax",
        "evidence",
        "span",
    ],
}

REPORT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "findings": {"type": "array", "items": FINDING_SCHEMA},
        "suvmax_all": {"type": "array", "items": {"type": "number"}},
        "numbers_salvaged": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "kind": {"type": "string", "enum": ["suvmax", "measurement"]},
                    "raw": {"type": "string"},
                    "value": {"type": "number"},
                    "unit": {"type": ["string", "null"]},
                    "section_hint": {"type": "string"},
                },
                "required": ["kind", "raw", "value", "section_hint"],
            },
        },
    },
    "required": ["findings"],
}

STRUCTURED_RESPONSE = {
    "type": "json_schema",
    "json_schema": {"name": "PSMAReport", "schema": REPORT_SCHEMA, "strict": True},
}

SYSTEM_MSG = (
    "You extract structured entities from cleaned PSMA PET/CT reports.\n"
    "Return STRICT JSON by the provided JSON Schema. Do not invent facts.\n"
    "Rules:\n"
    "- Sections: Prostatic fossa, Lymph nodes, Skeleton, Viscera (else Other/Unknown).\n"
    "- Negation/uncertainty policy: if status != 'present', set psma_status='n/a'.\n"
    "  e.g., 'No abnormal PSMA-positive lesion' or 'without abnormal PSMA-positive lesion' -> status='absent', psma_status='n/a', negation_cue='<matched phrase>'.\n"
    "- Associate each SUVmax and size with the most relevant local anatomy/site and section.\n"
    "- For each SUV, include raw_value_str (exact digits) and parsed float 'value' with comparator.\n"
    "- Provide a short verbatim 'evidence' (<=200 chars) and character span [start,end) over the original input string.\n"
    "- If a section states absence only, still include ONE 'absent' finding summarizing that section."
)

USER_TEMPLATE = """INPUT REPORT (cleaned):
---
{clean_text}
---
Return ONLY the JSON.
"""

FEWSHOT_TEMPLATE = """Example (input → JSON):
INPUT REPORT (cleaned):
---
{clean}
---
JSON:
{json}
"""

# ---------- helpers ----------
# SUV_CLAUSE = re.compile(r"(?i)\bSUV\s*max\b[^.\n;]*")
NUM_F = re.compile(r"[<>]?\s*\d+(?:\s*\.\s*\d+)?")
MEAS_RE = re.compile(
    r"(?ix)(\d+(?:\.\d+)?)\s*[x×]\s*(?:0?\.(\d+)|(\d+(?:\.\d+)?))\s*(mm|cm)\b"
)
NEG_ABSENT_RE = re.compile(  # (adds 'No PSMA-positive' and 'No PSMA avid/activity')
    r"(?i)\b(?:no\s+abnormal|without\s+abnormal|no\s+psma[-\s]*(?:positive|avid|activity))\b"
)

NEG_ANY_RE = re.compile(r"(?i)\b(?:no|there is no|without|absent)\b")
NEG_TARGET_RE = re.compile(r"(?i)\b(lesion|lymph node|nodule|metast|psma)\b")

NODE_ITEM_RE = re.compile(
    r"""
    (?:^|\r?\n|:\s)
    \s*
    (?:(?:cluster\s+of|a\s+couple\s+of|couple\s+of|a\s+few|few|several|multiple)\s+)?
    (?:(right|left)\s+)?              # optional leading laterality
    ([^,\n]+?)                        # site text up to next comma
    \s*(?:,\s*)?                     
    (?:with\s+the\s+index\s+)?        # optional
    (?:measuring\s*)?
    ([0-9.]+)\s*[x×]\s*([0-9.]+)\s*
    (mm|cm)
    \s*,\s*
    suv\s*max\s*=\s*([0-9.]+)
    """,
    re.IGNORECASE | re.VERBOSE | re.MULTILINE,
)
ORG_PATTERNS = [
    (r"\bpulmonary\b|\blungs?\b", "lung"),
    (r"\bthyroid\b", "thyroid"),
    (r"\bliver\b", "liver"),
    (r"\bgall ?bladder\b|\bgallstones?\b|cholelith", "gallbladder"),
    (r"\bkidney\b|\brenal\b", "kidney"),
    (r"\bspleen|\bsplenic\b", "spleen"),
    (r"\badrenal\b", "adrenal"),
    (r"\bpancreas|\bpancreatic\b", "pancreas"),
    (r"\bbladder\b", "bladder"),
    (r"\bpenile\b|\bpenis\b", "penile shaft"),
    (r"\bhernia\b", "hernia"),
    (r"\bhydrocele\b", "scrotum"),
    (r"\bsacral foramina?\b", "sacral foramina"),
]

INCIDENTAL_RE = re.compile(
    r"(?i)\b(physiologic(?:al)?|heterogeneous physiologic|non[- ]?specific|uncertain clinical significance|"
    r"most likely benign|felt less likely|likely inflammatory|inflammatory)\b"
)

# --- Invasion refinement (prevents leakage to lymph nodes etc.) ---
INV_TARGETS = [
    ("seminal vesicles", "seminal vesicle"),
    ("seminal vesicle", "seminal vesicle"),
    ("urinary bladder", "bladder"),
    ("bladder", "bladder"),
    ("rectal walls", "rectal wall"),
    ("rectal wall", "rectal wall"),
    ("rectum", "rectum"),
]
INV_TERMS_RE = re.compile(
    r"(?i)\b(invasion|invad(?:e|es|ed|ing)|extend(?:s|ed|ing|ion)|involv(?:e|es|ed|ing|ement))\b"
)

NEG_INV_RE = re.compile(
    r"(?i)\b(?:no|without|lack of|absent(?:\s+evidence\s+of)?)\s+"
    r"(?:invasion|extend(?:s|ed|ing|ion)|involv(?:e|es|ed|ing|ement))\b"
)


def map_site_from_text(txt: str) -> str:
    t = txt or ""
    for pat, name in ORG_PATTERNS:
        if re.search(pat, t):
            return name
    return "viscera"


def refine_other_and_viscera_sites(findings):
    for f in findings:
        if f.get("section") in ("Other", "Viscera"):
            ev = f.get("evidence", "")
            site = map_site_from_text(ev + " " + (f.get("site") or ""))
            if site and (f.get("site") in (None, "", "viscera", "unspecified")):
                f["site"] = site
            if site and (f.get("anatomy") in (None, "", "viscera", "unspecified")):
                f["anatomy"] = site


def strip_invades_outside_pf(findings):
    for f in findings:
        if f.get("section") != "Prostatic fossa":
            f["invades"] = []


def dedup_findings(findings):
    """Collapse exact-evidence duplicates; keep the more specific one."""

    def score(x):
        s = 0
        site = (x.get("site") or "").lower()
        if site not in {
            "",
            "viscera",
            "unspecified",
            "lymph nodes",
            "skeletal system",
            "prostate region",
        }:
            s += 3
        s += len(x.get("suvmax", []))
        if x.get("section") in ("Skeleton", "Other"):
            s += 1
        return s

    out, by_ev = [], {}
    for f in findings:
        ev_key = re.sub(r"\s+", " ", (f.get("evidence") or "").strip().lower())
        if not ev_key:
            out.append(f)
            continue
        g = by_ev.get(ev_key)
        if g is None:
            by_ev[ev_key] = f
            out.append(f)
        else:
            if score(f) > score(g):
                idx = out.index(g)
                out[idx] = f
                by_ev[ev_key] = f
    return out


def _canonicalize_inv_targets(text: str, section: Optional[str] = None) -> List[str]:
    """Return canonical invasion targets mentioned in text; skip if invasion is negated."""
    t = (text or "").lower().replace("vesicule", "vesicle")
    if not INV_TERMS_RE.search(t) or NEG_INV_RE.search(t):
        return []

    # per-section targets (PF gets SV; others inherit common targets)
    common = [
        ("urinary bladder", "bladder"),
        ("bladder", "bladder"),
        ("rectal walls", "rectal wall"),
        ("rectal wall", "rectal wall"),
        ("rectum", "rectum"),
    ]
    pf_only = [
        ("seminal vesicles", "seminal vesicle"),
        ("seminal vesicle", "seminal vesicle"),
    ]
    # (optional) skeleton extras; comment out if you prefer conservative behavior
    skeleton_extra = [
        ("spinal canal", "spinal canal"),
        ("epidural space", "epidural space"),
        ("neural foramen", "neural foramen"),
        ("neuroforamen", "neural foramen"),
        ("soft tissue", "soft tissue"),
        ("adjacent soft tissues", "soft tissue"),
    ]

    targets = list(common)
    if (section or "").lower() == "prostatic fossa":
        targets += pf_only
    elif (section or "").lower() == "skeleton":
        targets += skeleton_extra

    found = {canon for raw, canon in targets if raw in t}
    return sorted(found)


def _get_section_block(clean: str, section: str) -> tuple[str, int]:
    """
    Return (section_text, absolute_start_index_of_section_text_within_clean) or ("", -1)
    """
    pat = rf"(?is){re.escape(section)}\s*:(.*?)(?=Prostatic fossa:|Lymph nodes:|Skeleton:|Viscera:|\Z)"
    m = re.search(pat, clean)
    if not m:
        return "", -1
    return m.group(1), m.start(1)


def apply_invasion_refinement(parsed: Dict[str, Any], clean: str) -> None:
    pf_text, _ = _get_section_block(clean, "Prostatic fossa")
    for f in parsed.get("findings", []):
        # Only refine present findings. Do NOT clear others.
        if f.get("status") != "present":
            continue

        candidates = [
            f.get("evidence") or "",
            infer_from_window(clean, f.get("span", [0, 0]), f.get("site", "")),
        ]
        # PF gets the whole section as an extra context candidate
        if f.get("section") == "Prostatic fossa" and pf_text:
            candidates.append(pf_text or "")

        refined: List[str] = []
        for c in candidates:
            refined = _canonicalize_inv_targets(c, f.get("section"))
            if refined:
                break

        if refined:
            f["invades"] = refined
        # Note: if nothing refined, we **leave** whatever was already in f["invades"] (if any)


SENT_OR_LINE_SPLIT = re.compile(r"[;\n]+")

ORG_RE = re.compile(
    r"\b(thyroid|liver|kidney|spleen|adrenal|pancreas|lung|gallbladder|penile shaft)\b",
    re.I,
)


def explode_pf_bilateral(findings, clean):
    out = []
    pf_text, pf_abs = _get_section_block(clean, "Prostatic fossa")

    for f in findings:
        if f.get("section") != "Prostatic fossa" or f.get("status") != "present":
            out.append(f)
            continue

        if not pf_text:
            out.append(f)
            continue

        matches = list(
            re.finditer(
                r"(?i)\b(?:on\s+the\s+)?(left|right)\b[^.\n;]*?suv\s*max\s*=\s*([0-9.]+)",
                pf_text,
            )
        )
        if len(matches) < 2:
            out.append(f)
            continue

        # Split into one per side; drop the merged PF record
        for m in matches:
            side = m.group(1).lower()
            suv = m.group(2)
            ev = m.group(0).strip()
            s = pf_abs + m.start()
            out.append(
                {
                    "section": "Prostatic fossa",
                    "site": f"{side} prostate lobe",
                    "anatomy": "prostate",
                    "laterality": side,
                    "psma_status": "positive",
                    "status": "present",
                    "negation_cue": "",
                    "suvmax": [
                        {"raw_value_str": suv, "value": float(suv), "comparator": "="}
                    ],
                    "measurements": [],
                    "evidence": ev,
                    "span": [s, s + len(ev)] if s != -1 else [0, 0],
                    "invades": [],
                }
            )
    return out


def explode_skeleton_list(findings, clean):
    out = []
    sk_text, sk_abs = _get_section_block(clean, "Skeleton")

    for f in findings:
        if (
            f.get("section") != "Skeleton"
            or f.get("status") != "present"
            or not sk_text
        ):
            out.append(f)
            continue

        pieces = [p.strip() for p in SENT_OR_LINE_SPLIT.split(sk_text) if p.strip()]
        items = []
        for ev in pieces:
            m = re.search(r"(?i)\bsuv\s*max\s*=\s*([0-9.]+)", ev)
            if not m:
                continue
            suv = m.group(1)
            s = clean.find(ev, sk_abs)
            lat = infer_laterality_local(ev)
            site_text = ev.split(",")[0]
            site_text = re.sub(r"(?i)\b(left|right)\b", "", site_text).strip(" ,")
            items.append(
                {
                    "section": "Skeleton",
                    "site": site_text or "skeletal site",
                    "anatomy": "skeletal system",
                    "laterality": lat,
                    "psma_status": "positive",
                    "status": "present",
                    "negation_cue": "",
                    "suvmax": [
                        {"raw_value_str": suv, "value": float(suv), "comparator": "="}
                    ],
                    "measurements": [],
                    "evidence": ev,
                    "span": [s, s + len(ev)] if s != -1 else [0, 0],
                    "invades": [],
                }
            )
        if len(items) >= 2:
            out.extend(items)  # replace the merged one by split ones
        else:
            out.append(f)

    return out


def explode_viscera_sentences(findings, clean):
    out = []
    sec_text, sec_abs = _get_section_block(clean, "Viscera")
    out.extend([f for f in findings if f.get("section") != "Viscera"])

    if not sec_text:
        out.extend([f for f in findings if f.get("section") == "Viscera"])
        return out

    pieces = [p.strip() for p in SENT_OR_LINE_SPLIT.split(sec_text) if p.strip()]
    made_absent = False

    for ev in pieces:
        m_abs = NEG_ABSENT_RE.search(ev)
        if m_abs and not made_absent:
            s = clean.find(ev, sec_abs)
            out.append(
                {
                    "section": "Viscera",
                    "site": "viscera",
                    "anatomy": "viscera",
                    "laterality": None,
                    "psma_status": "n/a",
                    "status": "absent",
                    "negation_cue": m_abs.group(0),
                    "suvmax": [],
                    "measurements": [],
                    "evidence": ev,
                    "span": [s, s + len(ev)] if s != -1 else [0, 0],
                }
            )
            made_absent = True
            continue

        has_psma = re.search(r"\bpsma\b", ev, re.I)
        sec = "Viscera" if (has_psma and not INCIDENTAL_RE.search(ev)) else "Other"
        s = clean.find(ev, sec_abs)
        site = map_site_from_text(ev)

        out.append(
            {
                "section": sec,
                "site": site,
                "anatomy": site,
                "laterality": infer_laterality_local(ev),
                "psma_status": label_psma_status(ev) if sec == "Viscera" else "n/a",
                "status": "present",
                "negation_cue": "",
                "suvmax": [],
                "measurements": [],
                "evidence": ev,
                "span": [s, s + len(ev)] if s != -1 else [0, 0],
                "invades": [],
            }
        )
    return out


def explode_pf_seminal_vesicles(findings, clean):
    out = []
    pf_text, pf_abs = _get_section_block(clean, "Prostatic fossa")
    sv_zone = None
    if pf_text and re.search(r"\bseminal vesicle", pf_text, re.I):
        sv_zone = pf_text

    for f in findings:
        out.append(f)

    if not sv_zone:
        return out

    for m in re.finditer(
        r"(?i)\b(right|left)\b[^.\n]*?suv\s*max\s*=\s*([0-9.]+)", sv_zone
    ):
        side, suv = m.group(1).lower(), m.group(2)
        ev = m.group(0).strip()
        s = clean.find(ev, pf_abs)
        out.append(
            {
                "section": "Prostatic fossa",
                "site": f"{side} seminal vesicle",
                "anatomy": "seminal vesicle",
                "laterality": side,
                "psma_status": "positive",
                "status": "present",
                "negation_cue": "",
                "suvmax": [
                    {"raw_value_str": suv, "value": float(suv), "comparator": "="}
                ],
                "measurements": [],
                "evidence": ev,
                "span": [s, s + len(ev)] if s != -1 else [0, 0],
                "invades": [],
            }
        )
    return out


def explode_lymph_node_list(
    findings: List[Dict[str, Any]], clean: str
) -> List[Dict[str, Any]]:
    """
    Split multi-node Lymph node findings into separate findings by parsing the
    full Lymph nodes section (so we don't depend on the model's truncated evidence).
    """
    out: List[Dict[str, Any]] = []

    # Pull the whole section text once
    sec_text, sec_abs_start = _get_section_block(clean, "Lymph nodes")

    for f in findings:
        if f.get("section") != "Lymph nodes" or f.get("status") != "present":
            out.append(f)
            continue

        text_for_matching = sec_text if sec_text else (f.get("evidence") or "")
        matches = list(NODE_ITEM_RE.finditer(text_for_matching))

        # Only split when we clearly have a list (≥2 concrete items)
        if len(matches) < 2:
            out.append(f)
            continue

        for m in matches:
            lat = (m.group(1) or "").lower() or None
            site_rest = (m.group(2) or "").strip()
            # drop any preamble like "...as follows: "
            site_rest = re.sub(r".*:\s*", "", site_rest)

            # if laterality still missing, look inside the site text
            if not lat:
                mlat = re.search(r"\b(right|left)\b", site_rest, re.I)
                if mlat:
                    lat = mlat.group(1).lower()
                    # remove the laterality token from the site
                    site_rest = (
                        site_rest[: mlat.start()] + site_rest[mlat.end() :]
                    ).strip(" ,")

            a = float(m.group(3))
            b = float(m.group(4))
            unit = m.group(5)
            suv = m.group(6)

            raw = m.group(0)
            g_ev = raw.lstrip(" \r\n:")  # keep nice evidence text

            # Compute absolute span in `clean` robustly
            if sec_abs_start != -1 and text_for_matching is sec_text:
                leading_trim = len(raw) - len(g_ev)
                s = sec_abs_start + m.start() + leading_trim
            else:
                s = clean.find(g_ev)
            span = [s, s + len(g_ev)] if s != -1 else f.get("span", [0, 0])

            # Inherit everything except fields we explicitly replace
            g = {
                k: v
                for k, v in f.items()
                if k
                not in (
                    "suvmax",
                    "measurements",
                    "span",
                    "evidence",
                    "site",
                    "laterality",
                )
            }
            g["section"] = "Lymph nodes"
            g["site"] = f"{lat.title()} {site_rest}" if lat else site_rest
            g["laterality"] = lat
            g["suvmax"] = [
                {"raw_value_str": suv, "value": float(suv), "comparator": "="}
            ]
            g["measurements"] = [{"a": a, "b": b, "unit": unit}]
            g["evidence"] = g_ev
            g["span"] = span

            out.append(g)

        # (We intentionally do NOT append the original merged finding here)

    return out


def tail_after_last_section_offset(
    clean: str,
) -> Optional[int]:  # Reclassify text after the last section to “Other”
    m = re.search(r"(?is)(Prostatic fossa:|Lymph nodes:|Skeleton:|Viscera:)", clean)
    if not m:
        return None
    # find the last header
    anchors = [(m.start(), m.group(1))]
    for mm in re.finditer(
        r"(?is)(Prostatic fossa:|Lymph nodes:|Skeleton:|Viscera:)", clean
    ):
        anchors.append((mm.start(), mm.group(1)))
    last_pos = sorted(anchors)[-1][0]
    # look for a blank line separating the section body from a tail paragraph
    tail = re.search(r"\n\s*\n(.*)\Z", clean[last_pos:], re.S)
    if tail and tail.group(1).strip():
        return last_pos + tail.start(1)
    return None


def infer_laterality_local(txt: str) -> Optional[str]:
    t = (txt or "").lower()
    if re.search(r"\bbilateral\b|\bboth (lobes|sides|seminal vesicles)\b", t):
        return "bilateral"
    if re.search(r"\bleft\b", t):
        return "left"
    if re.search(r"\bright\b", t):
        return "right"
    return None


INV_KEYS = [
    "seminal vesicle",
    "seminal vesicles",
    "seminal vesicule",
    "bladder",
    "urinary bladder",
    "rectum",
    "rectal wall",
    "rectal walls",
]


def detect_invasion(txt: str):
    t = (txt or "").lower().replace("vesicule", "vesicle")  # normalize the typo
    # add a few verb forms20
    if any(
        w in t
        for w in [
            "invasion",
            "invading",
            "invade",
            "invades",
            "extension",
            "extending",
            "extends",
            "involvement",  # sometimes used instead of "invasion"
        ]
    ):
        return [k for k in INV_KEYS if k in t]
    return []


SUVMAX_VAL = re.compile(r"(?i)\bSUV\s*max\b[^0-9<>]*([<>]?\s*\d+(?:\s*\.\s*\d+)?)")


def salvage_numbers(clean_text: str) -> List[Dict[str, Any]]:
    out = []
    # SUVs (exact)
    for m in SUVMAX_VAL.finditer(clean_text):
        raw = m.group(1).strip()
        try:
            f = float(raw.replace("<", "").replace(">", "").replace(" ", ""))
            out.append(
                {
                    "kind": "suvmax",
                    "raw": raw,
                    "value": f,
                    "unit": None,
                    "section_hint": "Unknown",
                }
            )
        except:
            pass
    # Sizes (keep your existing MEAS_RE block as-is)
    for a, b_dec, b_full, unit in MEAS_RE.findall(clean_text):
        try:
            b = float(f"0.{b_dec}") if b_dec else float(b_full)
            out += [
                {
                    "kind": "measurement",
                    "raw": f"{a}x{b} {unit}",
                    "value": float(a),
                    "unit": unit,
                    "section_hint": "Unknown",
                },
                {
                    "kind": "measurement",
                    "raw": f"{a}x{b} {unit}",
                    "value": b,
                    "unit": unit,
                    "section_hint": "Unknown",
                },
            ]
        except:
            pass
    return out


def mm_from(a: float, unit: str) -> float:
    return float(a) * (10.0 if unit.lower() == "cm" else 1.0)


def deterministic_sample(
    rows: List[Dict[str, Any]], k: int, seed: int
) -> List[Dict[str, Any]]:
    rnd = random.Random(seed)
    pool = rows[: min(len(rows), 50)]
    rnd.shuffle(pool)
    return pool[: min(k, len(pool))]


def label_psma_status(text: str) -> str:
    t = text or ""
    if re.search(r"\bpsma[-\s]*negative\b", t, re.I):
        return "negative"
    if re.search(r"\bpsma[-\s]*positive\b", t, re.I):
        return "positive"
    if re.search(r"\bpsma[-\s]*(uptake|activity|avid|avidity|expression)\b", t, re.I):
        return "positive"
    if re.search(r"\bfoci\s+of\s+psma\b", t, re.I):
        return "positive"
    # PSMA mention near an SUV ⇒ treat as positive
    if re.search(r"\bpsma\b", t, re.I) and re.search(r"\bsuv(?:\s*max)?\b", t, re.I):
        return "positive"
    return "n/a"


def build_fewshot_blocks(
    all_rows: List[Dict[str, Any]], k: int = 3, seed: int = 13
) -> str:
    ex = deterministic_sample(all_rows, k, seed)
    blocks = []
    for row in ex:
        clean = row.get("clean", "")
        findings = []
        sections = row.get("sections", {})
        per_sec_key = {
            "Prostatic fossa": "suvmax_prostatic_fossa",
            "Lymph nodes": "suvmax_lymph_nodes",
            "Skeleton": "suvmax_skeleton",
            "Viscera": "suvmax_viscera",
        }
        for sec in SECTIONS:
            txt = sections.get(sec, "")
            sv = [
                {"raw_value_str": str(v), "value": float(v), "comparator": "="}
                for v in row.get(per_sec_key[sec], [])
            ]
            m = NEG_ABSENT_RE.search(txt)
            status = "absent" if m else ("present" if txt else "absent")
            psma_status = "n/a" if status != "present" else label_psma_status(txt)
            neg = m.group(0) if m else ""
            evidence = (txt[:200] or "").strip()
            s0 = clean.find(evidence) if evidence else 0
            s1 = s0 + len(evidence)
            site = {
                "Prostatic fossa": "prostate region",
                "Lymph nodes": "lymph nodes",
                "Skeleton": "skeletal system",
                "Viscera": "viscera",
            }[sec]
            findings.append(
                {
                    "section": sec,
                    "site": site,
                    "anatomy": site,
                    "laterality": None,
                    "psma_status": psma_status,
                    "status": status,
                    "negation_cue": neg,
                    "suvmax": sv,
                    "measurements": [],
                    "evidence": evidence,
                    "span": [max(0, s0), max(0, s1)],
                }
            )
        gold = {"findings": findings, "suvmax_all": row.get("suvmax_all", [])}
        blocks.append(
            FEWSHOT_TEMPLATE.format(
                clean=clean, json=json.dumps(gold, ensure_ascii=False)
            )
        )
    return "\n".join(blocks)


def exponential_backoff(fn, *args, max_tries=6, **kwargs):
    last_exc = None
    for attempt in range(max_tries):
        try:
            return fn(*args, **kwargs)
        except (APIConnectionError, RateLimitError, APIError) as e:
            last_exc = e
            sleep = (2**attempt) + random.random()
            time.sleep(sleep)
    # If we get here, we failed all attempts; do NOT return None.
    raise RuntimeError(f"API call failed after {max_tries} tries: {last_exc}")


def parse_structured_response_obj(resp) -> dict:
    # 1) Try parsed
    try:
        item = resp.output[0].content[0]
        parsed = getattr(item, "parsed", None)
        if parsed is not None:
            return parsed
    except Exception:
        pass
    # 2) Try text -> json
    for item in getattr(resp, "output", []) or []:
        for c in getattr(item, "content", []) or []:
            txt = getattr(c, "text", None)
            if txt:
                return json.loads(txt)
    # 3) Try whole-response text
    txt = getattr(resp, "output_text", None)
    if txt:
        return json.loads(txt)
    raise RuntimeError("Could not parse structured output from Responses API object.")


def extract_structured(
    client: OpenAI, model: str, clean_text: str, fewshot_block: str, seed: Optional[int]
) -> Dict[str, Any]:
    try:
        resp = exponential_backoff(
            client.responses.create,
            model=model,
            input=[
                {"role": "system", "content": SYSTEM_MSG},
                {
                    "role": "user",
                    "content": fewshot_block
                    + "\n"
                    + USER_TEMPLATE.format(clean_text=clean_text),
                },
            ],
            response_format=STRUCTURED_RESPONSE,
            # temperature=0,
            seed=seed,
            max_output_tokens=2200,
        )
        return parse_structured_response_obj(resp)
    except Exception:
        # Fallback to chat.completions JSON mode
        resp = exponential_backoff(
            client.chat.completions.create,
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_MSG},
                {
                    "role": "user",
                    "content": fewshot_block
                    + "\n"
                    + USER_TEMPLATE.format(clean_text=clean_text),
                },
            ],
            response_format={"type": "json_object"},
            # temperature=0,
            seed=seed,
        )
        if not getattr(resp, "choices", None):
            raise RuntimeError("chat.completions returned no choices.")
        txt = resp.choices[0].message.content
        if not isinstance(txt, str):
            raise RuntimeError("chat.completions content is empty (None).")
        return json.loads(txt)


def normalize_measurements_in_place(findings: List[Dict[str, Any]]):
    """
    Accepts measurements in any of these forms and normalizes them to dicts with mm:
      - dicts: {"a": 1.2, "b": 0.8, "unit": "cm"}
      - strings: "1.2 x 0.8 cm", "3.5 cm"
      - single dict, single string, or a list mixing any of the above
    For single-dimension values like "3.5 cm", we set b = a.
    """
    MEAS_ANY = re.compile(
        r"(?ix)(\d+(?:\.\d+)?)\s*(?:[x×]\s*(\d+(?:\.\d+)?))?\s*(mm|cm)\b"
    )

    def _coerce_list(mval):
        if mval is None:
            return []
        if isinstance(mval, list):
            return mval
        return [mval]  # wrap dict/str/number

    for f in findings:
        raw_meas = f.get("measurements", [])
        items = _coerce_list(raw_meas)
        fixed: List[Dict[str, Any]] = []

        for item in items:
            if isinstance(item, dict):
                unit = (item.get("unit") or "mm").lower()
                try:
                    a = float(item.get("a", 0.0))
                except Exception:
                    continue
                try:
                    b = float(item.get("b", a))  # if missing b, assume square
                except Exception:
                    b = a
                out = {"a": a, "b": b, "unit": unit}
            elif isinstance(item, (str, bytes)):
                m = MEAS_ANY.search(item.decode() if isinstance(item, bytes) else item)
                if not m:
                    continue
                a = float(m.group(1))
                b = float(m.group(2) or m.group(1))  # single-dim -> b = a
                unit = m.group(3).lower()
                out = {"a": a, "b": b, "unit": unit}
            else:
                # numbers or unknown types: skip
                continue

            out["mm"] = [mm_from(out["a"], out["unit"]), mm_from(out["b"], out["unit"])]
            fixed.append(out)

        f["measurements"] = fixed


def repair_evidence_and_spans(parsed: Dict[str, Any], clean: str):
    for f in parsed.get("findings", []):
        ev = f.get("evidence") or ""
        s, e = (f.get("span") or [0, 0])[:2]

        # strip "Description:" prefix if present
        ev2 = re.sub(r"(?i)^description:\s*", "", ev).strip()
        if ev2 != ev:
            f["evidence"] = ev2
            ev = ev2
            s2 = clean.find(ev2)
            if s2 != -1:
                f["span"] = [s2, s2 + len(ev2)]
                s, e = f["span"]

        # realign if mismatch
        if ev and (s < 0 or e <= s or clean[s:e] != ev):
            s2 = clean.find(ev)
            if s2 != -1:
                f["span"] = [s2, s2 + len(ev)]
            else:
                f["span"] = [max(0, s), max(0, s) + len(ev)]
        elif not ev and 0 <= s < e <= len(clean):
            f["evidence"] = clean[s:e]


def infer_from_window(
    clean: str, span: List[int], site: str
) -> (
    str
):  # Infer laterality and invasion from the source text, not the trimmed evidence
    s, e = (span + [0, 0])[:2]
    w = clean[max(0, s - 200) : min(len(clean), e + 200)]
    return w + " " + (site or "")


def _maybe_flip_to_absent(
    f,
):  # safety net that flips present → absent when the evidence looks negated
    # We handled ("No abnormal" / "without abnormal") but not ("No PSMA-positive ...")
    ev = f.get("evidence") or ""
    if (
        f.get("status") == "present"
        and NEG_ANY_RE.search(ev)
        and NEG_TARGET_RE.search(ev)
    ):
        # capture a short cue
        m = re.search(r"(?i)\b(?:no|there is no|without|absent)\b[^.\n;]*", ev)
        f["status"] = "absent"
        f["psma_status"] = "n/a"
        f["negation_cue"] = m.group(0) if m else "negated"


def post_validate_and_fix(parsed: Dict[str, Any], clean: str) -> Dict[str, Any]:
    # attach salvage if missing
    if "numbers_salvaged" not in parsed or not isinstance(
        parsed["numbers_salvaged"], list
    ):
        parsed["numbers_salvaged"] = salvage_numbers(clean)

    # minimal schema repair then validate
    if "findings" not in parsed or not isinstance(parsed["findings"], list):
        parsed["findings"] = []

    # Ensure absent placeholders per missing section
    present_secs = {f.get("section") for f in parsed["findings"] if isinstance(f, dict)}
    for sec in SECTIONS:
        if sec not in present_secs and re.search(rf"(?is){re.escape(sec)}\s*:", clean):
            # try to detect absence string
            sec_block = re.search(
                rf"(?is){re.escape(sec)}\s*:(.*?)(?:Prostatic fossa:|Lymph nodes:|Skeleton:|Viscera:|\Z)",
                clean,
            )
            txt = sec_block.group(1).strip() if sec_block else ""
            m_neg = NEG_ABSENT_RE.search(txt)
            is_absent = bool(m_neg or re.search(r"(?i)\bNo\b[^.\n;]*\blesion\b", txt))
            if is_absent:
                cue = m_neg.group(0) if m_neg else "No lesion"
                evidence = (txt[:200] or "").strip()
                s0 = clean.find(evidence) if evidence else 0
                s1 = s0 + len(evidence)
                parsed["findings"].append(
                    {
                        "section": sec,
                        "site": sec.lower(),
                        "anatomy": sec.lower(),
                        "laterality": None,
                        "psma_status": "n/a",
                        "status": "absent",
                        "negation_cue": cue,
                        "suvmax": [],
                        "measurements": [],
                        "evidence": evidence,
                        "span": [max(0, s0), max(0, s1)],
                    }
                )

    # Fill required defaults
    for f in list(parsed["findings"]):
        if not isinstance(f, dict):
            parsed["findings"].remove(f)
            continue
        # fixing Treat “PSMA activity” as positive (unless explicitly negative)
        if f.get("status") == "present":
            f["psma_status"] = label_psma_status(f.get("evidence", ""))
        else:
            f["psma_status"] = "n/a"
        #################################################################
        f.setdefault("site", "unspecified")
        f.setdefault("psma_status", "n/a")
        f.setdefault("status", "present")
        f.setdefault("negation_cue", "")
        f.setdefault("suvmax", [])
        f.setdefault("measurements", [])
        f.setdefault("laterality", None)
        f.setdefault("section", "Unknown")
        f.setdefault("evidence", "")
        f.setdefault("span", [0, min(200, len(clean))])

        # laterality
        if f.get("status") == "present":
            lat = infer_laterality_local(
                (f.get("evidence", "") or "") + " " + (f.get("site", "") or "")
            )
            f["laterality"] = lat
        else:
            f["laterality"] = None
        if not f.get("invades"):
            inv_src = infer_from_window(clean, f.get("span", [0, 0]), f.get("site", ""))
            inv = detect_invasion(inv_src)
            if inv:
                f["invades"] = inv
        suv_fixed = []
        for s in f.get("suvmax", []):
            if isinstance(s, dict):
                if "raw_value_str" not in s and "value" in s:
                    s["raw_value_str"] = str(s["value"])
                s.setdefault("comparator", "=")
                suv_fixed.append(s)
            else:
                # string/number -> make a proper object
                raw = str(s)
                try:
                    # trim trailing non-numeric punctuation (e.g., "10.5.")
                    val = float(re.sub(r"[^\d.+-]+$", "", raw))
                    suv_fixed.append(
                        {"raw_value_str": raw, "value": val, "comparator": "="}
                    )
                except Exception:
                    continue
        f["suvmax"] = suv_fixed

        _maybe_flip_to_absent(f)

    normalize_measurements_in_place(parsed["findings"])
    repair_evidence_and_spans(parsed, clean)

    parsed["findings"] = explode_lymph_node_list(parsed["findings"], clean)
    # Split multi-node paragraphs into separate findings
    parsed["findings"] = explode_pf_seminal_vesicles(parsed["findings"], clean)
    parsed["findings"] = explode_pf_bilateral(parsed["findings"], clean)
    parsed["findings"] = explode_skeleton_list(parsed["findings"], clean)
    parsed["findings"] = explode_viscera_sentences(parsed["findings"], clean)

    # only PF is allowed to carry "invades"
    strip_invades_outside_pf(parsed["findings"])

    apply_invasion_refinement(parsed, clean)

    # Reclassify text after the last section to “Other”
    tail = tail_after_last_section_offset(clean)
    if tail is not None:
        for f in parsed["findings"]:
            if (f.get("span") or [0])[0] >= tail:
                f["section"] = "Other"

    refine_other_and_viscera_sites(parsed["findings"])
    parsed["findings"] = dedup_findings(parsed["findings"])

    # Final validation
    try:
        validate(instance=parsed, schema=REPORT_SCHEMA)
    except ValidationError as e:
        # If still failing, keep only minimally valid fields
        ok_findings = []
        for f in parsed.get("findings", []):
            try:
                validate(instance={"findings": [f]}, schema=REPORT_SCHEMA)
                ok_findings.append(f)
            except ValidationError:
                continue
        parsed = {
            "findings": ok_findings,
            "suvmax_all": parsed.get("suvmax_all", []),
            "numbers_salvaged": parsed.get("numbers_salvaged", []),
        }
        validate(instance=parsed, schema=REPORT_SCHEMA)

    return parsed


def finding_to_descriptor(f: Dict[str, Any]) -> str:
    suv_str = (
        ",".join(
            [
                f'{x.get("comparator","=")}{x.get("value"):.3f}'
                for x in f.get("suvmax", [])
            ]
        )
        or "none"
    )
    meas = f.get("measurements", [])
    meas_str = ";".join([f'{m["a"]}x{m["b"]}{m["unit"]}' for m in meas]) or "none"
    cue = f.get("negation_cue", "")
    lat = f.get("laterality") or "unspecified"
    inv = ",".join(f.get("invades", []) or []) or "none"
    return (
        f'section={f.get("section","")} | site={f.get("site","")} | status={f.get("status","")} '
        f'| psma={f.get("psma_status","n/a")} | lat={lat} | invades={inv} | SUVmax={suv_str} '
        f'| meas={meas_str} | cue={cue} | evidence="{(f.get("evidence","") or "")[:100]}"'
    )


def finding_to_phrase(f: Dict[str, Any]) -> List[str]:
    out = []
    site = f.get("site", "")
    sec = f.get("section", "Unknown")
    stat = f.get("status", "present")
    psma = f.get("psma_status", "n/a")
    ev = (f.get("evidence", "") or "")[:140]
    out.append(f"[{sec}] {site} | {stat} | PSMA-{psma} :: {ev}")
    for s in f.get("suvmax", []):
        out.append(
            f"SUVmax {s.get('comparator','=')}{s.get('raw_value_str', str(s.get('value','')))}"
        )
    for m in f.get("measurements", []):
        out.append(
            "size " + "×".join(str(x) for x in m.get("mm", [m["a"], m["b"]])) + " mm"
        )
    return [x for x in out if x]


def embed_strings(client: OpenAI, model: str, strings: List[str]) -> List[List[float]]:
    if not strings:
        return []
    vecs: List[List[float]] = []
    CHUNK = 96
    for i in range(0, len(strings), CHUNK):
        batch = strings[i : i + CHUNK]
        r = exponential_backoff(client.embeddings.create, model=model, input=batch)
        vecs.extend([d.embedding for d in r.data])
    return vecs


def load_clean_records(path: Path) -> List[Dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_clean_jsonl", required=True)
    ap.add_argument("--out_dir", default="cache/text")
    ap.add_argument("--extract_model", default="gpt-5-mini")
    ap.add_argument("--embed_model", default="text-embedding-3-small")
    ap.add_argument("--max_examples", type=int, default=3)
    ap.add_argument(
        "--embed_mode", choices=["descs", "phrases", "both"], default="both"
    )
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    random.seed(args.seed)
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # uses OPENAI_API_KEY

    inp = Path(args.in_clean_jsonl)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "index_gpt.jsonl"

    rows = load_clean_records(inp)
    fewshot_block = build_fewshot_blocks(rows, k=args.max_examples, seed=args.seed)

    n = 0
    with open(index_path, "w", encoding="utf-8") as findex:
        for row in rows:
            rid = row.get("ID") or row.get("id")
            clean = (row.get("clean") or row.get("Description") or "").strip()
            if not rid or not clean:
                continue

            # 1) extraction
            parsed = extract_structured(
                client, args.extract_model, clean, fewshot_block, seed=args.seed
            )
            parsed = post_validate_and_fix(parsed, clean)

            # 2) memory strings to embed
            descs: List[str] = []
            phrases: List[str] = []
            for f in parsed["findings"]:
                #################################
                # ev = f.get("evidence", "")
                # if f.get("status") != "present":
                #     f["psma_status"] = "n/a"
                # else:
                #     if f.get("psma_status") in (None, "", "n/a"):
                #         f["psma_status"] = label_psma_status(ev)
                #################################
                if args.embed_mode in ("descs", "both"):
                    descs.append(finding_to_descriptor(f))
                if args.embed_mode in ("phrases", "both"):
                    phrases.extend(finding_to_phrase(f))

            # also pooled descriptors per section (helps stability)
            if args.embed_mode in ("descs", "both"):
                by_sec = {s: [] for s in SECTIONS}
                for d, f in zip(descs, parsed["findings"]):
                    s = f.get("section", "Unknown")
                    if s in by_sec:
                        by_sec[s].append(d)
                for s in SECTIONS:
                    if by_sec[s]:
                        descs.append(f"[SECTION:{s}] " + " || ".join(by_sec[s])[:1200])

            # dedup & clip
            mem_strings = []
            if args.embed_mode in ("descs", "both"):
                mem_strings.extend(descs)
            if args.embed_mode in ("phrases", "both"):
                mem_strings.extend(phrases)
            mem_strings = [s.strip() for s in mem_strings if s and s.strip()]
            mem_strings = list(dict.fromkeys(mem_strings))  # dedup keep order

            # 3) embeddings
            vecs = embed_strings(client, args.embed_model, mem_strings)
            E = (
                torch.tensor(np.array(vecs, dtype="float32"))
                if vecs
                else torch.empty(0, 0)
            )
            H = int(E.shape[-1]) if E.numel() else 0

            # 4) write artifacts
            meta_path = out_dir / f"{rid}_gpt.json"
            torch_path = out_dir / f"{rid}_gpt_entities.pt"

            meta = {
                "id": rid,
                "extract_model": args.extract_model,
                "embed_model": args.embed_model,
                "num_findings": len(parsed.get("findings", [])),
                "hidden_dim": H,
                "findings": parsed.get("findings", []),
                "suvmax_all": parsed.get("suvmax_all", []),
                "numbers_salvaged": parsed.get("numbers_salvaged", []),
                "embed_mode": args.embed_mode,
                "num_entities": len(mem_strings),
            }
            with open(meta_path, "w", encoding="utf-8") as fjson:
                json.dump(meta, fjson, ensure_ascii=False, indent=2)

            torch.save({"strings": mem_strings, "embeddings": E}, torch_path)

            findex.write(
                json.dumps(
                    {
                        "id": rid,
                        "clean_json": str(meta_path),
                        "entities_pt": str(torch_path),
                        "hidden_dim": H,
                        "num_entities": len(mem_strings),
                    }
                )
                + "\n"
            )

            n += 1

    print(f"Processed {n} reports → {out_dir} (index: {index_path})")


if __name__ == "__main__":
    main()
