"""
python psma_models/tools/viz_msraw.py \
  --id 135 \
  --msraw_cache ./cache_msraw \
  --attn_dir ./runs/msraw/test_fold0/attn_tokens \
  --clean_jsonl ./cache_msraw/clean_reports.jsonl \
  --topk 25 --dump_tsv /tmp/135_msraw_sent_attn.tsv
"""

import argparse, json, re
from pathlib import Path
from typing import List, Tuple
import numpy as np
import torch
from transformers import BertTokenizerFast

SECTION_HDR = re.compile(
    r"(?im)^(Prostatic fossa|Lymph nodes|Skeleton|Viscera)\s*:\s*$"
)
SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])|\n+")  # sentence-ish


def sentence_spans(text: str) -> List[Tuple[int, int]]:
    bounds = [0]
    for m in SPLIT_PATTERN.finditer(text):
        bounds.append(m.end())
    bounds.append(len(text))
    spans = []
    for i in range(len(bounds) - 1):
        a, b = bounds[i], bounds[i + 1]
        seg = text[a:b]
        a += len(seg) - len(seg.lstrip())
        b -= len(seg) - len(seg.rstrip())
        if a < b:
            spans.append((a, b))
    return spans


def find_sections(text: str) -> List[Tuple[int, int, str]]:
    spans = []
    pos = 0
    for ln in text.split("\n"):
        st, en = pos, pos + len(ln)
        if SECTION_HDR.match(ln.strip()):
            spans.append((st, en, ln.strip()))
        pos = en + 1
    return spans


def section_for(char_start: int, sec_spans) -> str:
    hdr = ""
    for hs, he, h in sec_spans:
        if hs <= char_start:
            hdr = h
        else:
            break
    return hdr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument("--msraw_cache", required=True, help="dir with *_msraw_tokens.pt")
    ap.add_argument(
        "--attn_dir", required=True, help="runs/<exp>/<split>_foldk/attn_tokens"
    )
    ap.add_argument("--clean_jsonl", required=True)
    ap.add_argument("--hf_id", default="microsoft/BiomedVLP-CXR-BERT-specialized")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--dump_tsv", default="")
    args = ap.parse_args()

    rid = str(int(args.id))
    p_pt = Path(args.msraw_cache) / f"{int(rid):04d}_msraw_tokens.pt"
    p_meta = Path(args.msraw_cache) / f"{rid}_msraw.json"
    p_attn = Path(args.attn_dir) / f"{rid}_token_scores.npy"
    if not p_pt.exists() or not p_attn.exists():
        raise SystemExit(f"Missing file(s): {p_pt} or {p_attn}")

    pack = torch.load(p_pt, map_location="cpu")
    tokens: torch.Tensor = pack.get("tokens", torch.empty(0, 0))
    mask: torch.Tensor = pack.get("mask", torch.empty(0, dtype=torch.bool))
    L = int(tokens.shape[0])
    scores = np.load(p_attn)
    if L == 0 or scores.size == 0:
        raise SystemExit("No tokens or scores.")

    # load raw report text
    text = None
    with open(args.clean_jsonl, "r", encoding="utf-8") as f:
        for ln in f:
            j = json.loads(ln)
            if str(j.get("ID") or j.get("id")) == str(f"{int(rid):04d}"):
                text = (j.get("clean") or j.get("Description") or "").strip()
                break
    if not text:
        raise SystemExit("Could not find clean text for this ID.")

    # re-tokenize to get offset mapping (no special tokens)
    tok = BertTokenizerFast.from_pretrained(
        args.hf_id, use_fast=True, trust_remote_code=True
    )
    enc = tok(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_attention_mask=True,
        truncation=False,
    )
    offsets = [
        (a, b)
        for (a, b), m in zip(
            enc["offset_mapping"],
            enc.get("attention_mask", [1] * len(enc["offset_mapping"])),
        )
        if m == 1 and (b - a) > 0
    ]

    # align lengths if needed
    L2 = min(L, len(offsets), scores.size)
    offsets = offsets[:L2]
    scores = scores[:L2]

    # sentence spans + quick section headers
    sents = sentence_spans(text)
    secs = find_sections(text)

    # map sentence -> token idx span by offset overlap
    rows = []
    cur = 0
    for si, (sa, sb) in enumerate(sents):
        start_t = None
        end_t = None
        for ti in range(cur, L2):
            a, b = offsets[ti]
            if b <= sa:
                continue
            if a >= sb:
                break
            if start_t is None:
                start_t = ti
            end_t = ti
        if start_t is None:
            continue
        cur = end_t + 1
        attn_sum = float(scores[start_t : end_t + 1].sum())
        attn_mean = float(scores[start_t : end_t + 1].mean())
        seg = re.sub(r"\s+", " ", text[sa:sb]).strip()
        rows.append(
            {
                "sent_i": si,
                "tok_start": start_t,
                "tok_end": end_t,
                "tok_count": end_t - start_t + 1,
                "attn_sum": attn_sum,
                "attn_mean": attn_mean,
                "section": section_for(sa, secs),
                "preview": seg[:160],
            }
        )

    rows.sort(key=lambda r: r["attn_sum"], reverse=True)
    topk = min(args.topk, len(rows))
    print(f"\n=== MS-RAW sentence attention — ID {rid} (top {topk} by sum) ===")
    for r in rows[:topk]:
        sec = f"[{r['section']}] " if r["section"] else ""
        print(
            f"[{r['sent_i']:03d}] sum={r['attn_sum']:.4f} mean={r['attn_mean']:.4f} "
            f"toks=({r['tok_start']},{r['tok_end']}) n={r['tok_count']:3d} | {sec}{r['preview']}"
        )

    # optional: aggregate by section
    from collections import defaultdict

    by_sec = defaultdict(float)
    for r in rows:
        by_sec[r["section"]] += r["attn_sum"]
    if by_sec:
        print("\n-- Section totals (attn_sum) --")
        for k, v in sorted(by_sec.items(), key=lambda kv: kv[1], reverse=True):
            name = k or "(no header)"
            print(f"{name:20s} : {v:.4f}")

    if args.dump_tsv:
        import csv

        with open(args.dump_tsv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"\nSaved table → {args.dump_tsv}\n")


if __name__ == "__main__":
    main()
