"""
Usage:
python docx_to_text.py --docx_root /path/to/reports --out_jsonl cache/text/raw_reports.jsonl
"""

# coding: utf-8

import argparse
import json
import unicodedata
from pathlib import Path
from typing import Iterable
import os
import csv
from dotenv import load_dotenv

load_dotenv()

CSV_MAPPING = Path(os.environ["PATIENTS_MAPPING_DIR"])

patient_mapping = {}  # names as keys, IDs as values
with open(CSV_MAPPING, newline="", encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):
        patient_mapping[row["original"].strip()] = row["anon_id"].strip()

try:
    from docx import Document  # python-docx
except Exception as e:
    raise SystemExit("Please install python-docx:  pip install python-docx") from e


def read_docx_keep_lines(path: Path) -> str:
    doc = Document(str(path))
    collecting = False
    lines = []
    for para in doc.paragraphs:
        t = para.text.strip() or ""
        t = unicodedata.normalize("NFKC", t.rstrip("\r\n"))
        if not collecting and t.lower().startswith("description:"):
            collecting = True  # start keeping lines from here
        if collecting:
            lines.append(t)
    txt = "\n".join(lines)
    # Normalize newlines and some common glyphs without touching \n
    txt = (
        txt.replace("\r\n", "\n").replace("\r", "\n").replace("\u00d7", "x")
    )  # Replaces the Unicode multiplication sign “×” (U+00D7) with the ASCII letter “x”.
    return txt


def extract_patient_name(path: Path) -> str:
    p = (
        str(path).split("\\")[-4].split("-")[0]
    )  # e.g John_Smith_01-11-26-10 ---> John_Smith_01
    # There is only one example that goes like John_Smith and not like John_Smith_01

    if len(p.split("_")) == 2:
        return p
    else:
        return p[:-3]


def iter_docx(root: Path) -> Iterable[Path]:
    # Recursive scan
    yield from root.rglob("*.docx")


def main():
    ap = argparse.ArgumentParser(description="DOCX → raw_reports.jsonl")
    ap.add_argument(
        "--docx_root",
        type=str,
        required=True,
        help="Directory containing .docx reports (scanned recursively).",
    )
    ap.add_argument(
        "--out_jsonl",
        type=str,
        required=True,
        help="Output JSONL path, e.g., cache/text/raw_reports.jsonl",
    )
    args = ap.parse_args()

    root = Path(args.docx_root)
    outp = Path(args.out_jsonl)
    outp.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with outp.open("w", encoding="utf-8") as f:
        for fp in iter_docx(root):
            try:
                raw = read_docx_keep_lines(fp)
                patient_name = extract_patient_name(fp)
                rec = {"ID": patient_mapping[patient_name], "Description": raw}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
            except Exception as e:
                print(f"[WARN] Skipping {fp}: {e}")

    print(f"Wrote {n} records → {outp}")


if __name__ == "__main__":
    main()
