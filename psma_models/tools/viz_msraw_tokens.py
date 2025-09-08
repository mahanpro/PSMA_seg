#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize MS-RAW token embeddings grouped by SENTENCES (no re-embedding).
It aligns the saved token embeddings to sentence spans using offset_mapping.

Usage:
python viz_msraw_by_sentence.py \
  --clean_jsonl /path/to/clean_reports.jsonl \
  --out_dir /path/to/cache_msraw \
  --id 0024 \
  --hf_id microsoft/BiomedVLP-CXR-BERT-specialized \
  --drop_special \
  --max_preview_chars 140 \
  --dump_tsv /tmp/0024_msraw_sentences.tsv
"""
import argparse, json, re
from pathlib import Path
from typing import List, Optional, Tuple
import torch
import numpy as np

from transformers import BertTokenizerFast

SECTION_HDR = re.compile(
    r"(?im)^(Prostatic fossa|Lymph nodes|Skeleton|Viscera)\s*:\s*$"
)

# sentence/newline boundaries: end of . ! ? followed by space + capital/(
# OR any run of newlines
SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])|\n+")


def load_clean(path: Path, rid: str) -> Optional[str]:
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            if not ln.strip():
                continue
            obj = json.loads(ln)
            if str(obj.get("ID") or obj.get("id")) == rid:
                return (obj.get("clean") or obj.get("Description") or "").strip()
    return None


def find_sections(text: str) -> List[Tuple[int, int, str]]:
    """Return list of (start_char, end_char, header_line) for section headers present in text."""
    spans = []
    start = 0
    lines = text.split("\n")
    pos = 0
    for ln in lines:
        ln_stripped = ln.strip()
        st = pos
        en = pos + len(ln)
        if SECTION_HDR.match(ln_stripped):
            spans.append((st, en, ln_stripped))
        pos = en + 1  # + newline
    return spans


def sentence_spans(text: str) -> List[Tuple[int, int]]:
    """Return character spans for sentences/lines based on SPLIT_PATTERN."""
    bounds = [0]
    for m in SPLIT_PATTERN.finditer(text):
        bounds.append(m.end())
    bounds.append(len(text))
    spans = []
    for i in range(len(bounds) - 1):
        a, b = bounds[i], bounds[i + 1]
        # trim whitespace at edges but preserve original indices when possible
        seg = text[a:b]
        left_ws = len(seg) - len(seg.lstrip())
        right_ws = len(seg) - len(seg.rstrip())
        a2 = a + left_ws
        b2 = b - right_ws
        if a2 < b2:
            spans.append((a2, b2))
    return spans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean_jsonl", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--id", required=True)
    ap.add_argument("--hf_id", default="microsoft/BiomedVLP-CXR-BERT-specialized")
    ap.add_argument("--drop_special", action="store_true")
    ap.add_argument("--max_preview_chars", type=int, default=140)
    ap.add_argument("--dump_tsv", default="")
    args = ap.parse_args()

    rid = args.id
    out_dir = Path(args.out_dir)
    pt_path = out_dir / f"{rid}_msraw_tokens.pt"
    meta_path = out_dir / f"{rid}_msraw.json"

    print("\n========== MS-RAW SENTENCE VIS (verbose) ==========")
    print(f"ID: {rid}")
    print(f"Paths:\n  PT:   {pt_path}\n  META: {meta_path}\n  CLEAN:{args.clean_jsonl}")

    if not pt_path.exists():
        print(f"[ERROR] Missing PT file: {pt_path}")
        return

    # Load token embeddings and mask saved by msraw_embed.py
    try:
        data = torch.load(pt_path, map_location="cpu")
    except Exception as e:
        print(f"[ERROR] Failed to load {pt_path}: {e}")
        return

    tokens: torch.Tensor = data.get("tokens", torch.empty(0, 0))
    mask: torch.Tensor = data.get("mask", torch.empty(0, dtype=torch.bool))
    L, Ct = (int(tokens.shape[0]), int(tokens.shape[1]) if tokens.ndim == 2 else 0)

    meta = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}

    print(
        f"Loaded: tokens_shape={tuple(tokens.shape)} dtype={getattr(tokens,'dtype',None)}  mask_true={int(mask.sum().item()) if mask.numel() else 0}"
    )
    if meta:
        print(f"Meta says: {meta}")

    # Load raw text
    txt = load_clean(Path(args.clean_jsonl), rid)
    if not txt:
        print(f"[ERROR] Could not load clean text for ID={rid}.")
        return
    print(f"Report length: chars={len(txt):,}")

    # Tokenize raw text to get offset mapping **without** specials,
    # so number of offsets should match L when --drop_special was used.
    try:
        tok = BertTokenizerFast.from_pretrained(
            args.hf_id, use_fast=True, trust_remote_code=True
        )
    except Exception as e:
        print(f"[ERROR] Failed to load tokenizer {args.hf_id}: {e}")
        return

    enc = tok(
        txt,
        add_special_tokens=False,  # align with --drop_special
        return_offsets_mapping=True,
        return_attention_mask=True,
        truncation=False,
    )
    offsets = enc["offset_mapping"]  # List[(start,end)]
    attn = enc.get("attention_mask", [1] * len(offsets))
    # Filter out tokens with zero-length offsets or masked-out tokens
    valid = [
        (i, o)
        for i, (o, a) in enumerate(zip(offsets, attn))
        if a == 1 and (o[1] - o[0]) > 0
    ]
    tok_idx = [i for i, _ in valid]
    tok_spans = [o for _, o in valid]

    if len(tok_idx) != L:
        print(
            f"[WARN] Tokenizer produced {len(tok_idx)} offsets but saved embeddings have L={L}."
        )
        print(
            "      Proceeding with min length alignment; some boundaries may be approximate."
        )
    align_L = min(L, len(tok_idx))

    # Find sentences (character spans)
    sent_sp = sentence_spans(txt)
    sec_spans = find_sections(txt)

    # For quick section lookup, for each sentence pick the last header whose start <= sentence.start
    def section_for(a_char: int) -> Optional[str]:
        header = None
        for hs, he, htxt in sec_spans:
            if hs <= a_char:
                header = htxt
            else:
                break
        return header

    # Map sentences -> token index ranges by offset overlap
    rows = []
    cur_tok = 0
    for si, (sa, sb) in enumerate(sent_sp):
        # Collect tokens whose offsets intersect [sa,sb)
        start_tok = None
        end_tok = None
        for ti in range(cur_tok, align_L):
            a, b = tok_spans[ti]
            if b <= sa:
                continue
            if a >= sb:
                break
            if start_tok is None:
                start_tok = ti
            end_tok = ti
        if start_tok is None or end_tok is None:
            continue
        # Advance cursor to next unseen token
        cur_tok = end_tok + 1

        # Compute sentence stats and preview
        seg = txt[sa:sb]
        seg_compact = re.sub(r"\s+", " ", seg).strip()
        tcount = end_tok - start_tok + 1

        # Mean embedding for the sentence span (optional)
        if start_tok < L and end_tok < L and tokens.ndim == 2 and L > 0:
            seg_emb = tokens[start_tok : end_tok + 1].mean(dim=0)
            seg_norm = float(torch.linalg.norm(seg_emb).item())
        else:
            seg_norm = float("nan")

        rows.append(
            {
                "sent_i": si,
                "char_start": sa,
                "char_end": sb,
                "tok_start": start_tok,
                "tok_end": end_tok,
                "tok_count": tcount,
                "section": section_for(sa) or "",
                "emb_norm_mean": seg_norm,
                "preview": seg_compact[: args.max_preview_chars],
            }
        )

    # Print summary
    toks_per_sent = [r["tok_count"] for r in rows]
    if toks_per_sent:
        tp = np.array(toks_per_sent, dtype=np.int32)
        print(f"\nSentences found: {len(rows)}")
        print(
            f"Tokens per sentence: min={tp.min()}  mean={tp.mean():.2f}  max={tp.max()}"
        )
    else:
        print(
            "\n[WARN] No sentence rows produced (check tokenizer offsets and splitting)."
        )

    print("\n-- Sentences --")
    for r in rows:
        sec = f"[{r['section']}] " if r["section"] else ""
        print(
            f"[{r['sent_i']:03d}] chars=({r['char_start']:4d},{r['char_end']:4d}) "
            f"toks=({r['tok_start']:4d},{r['tok_end']:4d}) n={r['tok_count']:3d} "
            f"|‖mean(E)‖={r['emb_norm_mean']:.3f} | {sec}{r['preview']}"
        )

    # Optional TSV dump
    if args.dump_tsv:
        import csv

        with open(args.dump_tsv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"\nSaved per-sentence table → {args.dump_tsv}")

    print("==============================================\n")


if __name__ == "__main__":
    main()
