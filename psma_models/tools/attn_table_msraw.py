"""
python psma_models/tools/attn_table_msraw.py \
  --id 135 \
  --msraw_cache ./cache_msraw \
  --attn_dir   ./runs/msraw/test_fold0/attn_tokens \
  --clean_jsonl ./clean_reports.jsonl \
  --out_tsv ./runs/msraw/test_fold0/msraw.tsv

"""

import argparse, json, re
from pathlib import Path
import numpy as np
import torch
from transformers import BertTokenizerFast


def load_clean_text(clean_jsonl: Path, rid: str) -> str:
    rid = str(int(rid))
    with open(clean_jsonl, "r", encoding="utf-8") as f:
        for ln in f:
            j = json.loads(ln)
            jid = str(j.get("ID") or j.get("id") or "")
            if jid.lstrip("0") == rid.lstrip("0"):
                return (j.get("clean") or j.get("Description") or "").strip()
    raise SystemExit(f"Could not find clean text for ID={rid} in {clean_jsonl}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument("--msraw_cache", required=True, help="dir with *_msraw_tokens.pt")
    ap.add_argument("--attn_dir", required=True, help="dir with *_token_scores.npy")
    ap.add_argument("--clean_jsonl", required=True)
    ap.add_argument("--hf_id", default="microsoft/BiomedVLP-CXR-BERT-specialized")
    ap.add_argument("--out_tsv", required=True)
    args = ap.parse_args()

    rid_z = f"{int(args.id):04d}"
    tokens_pt = Path(args.msraw_cache) / f"{rid_z}_msraw_tokens.pt"
    scores_npy = Path(args.attn_dir) / f"{int(args.id)}_token_scores.npy"

    if not tokens_pt.exists() or not scores_npy.exists():
        raise SystemExit(f"Missing: {tokens_pt} or {scores_npy}")

    # Load scores (L,)
    scores = np.load(scores_npy).astype(np.float64)

    # Re-tokenize the CLEAN TEXT with the same HF model to get offsets
    text = load_clean_text(Path(args.clean_jsonl), args.id)
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
    offs = [
        (a, b)
        for (a, b), m in zip(enc["offset_mapping"], enc.get("attention_mask", []))
        if m == 1 and (b - a) > 0
    ]
    toks = tok.convert_ids_to_tokens(enc["input_ids"])

    L = min(len(offs), len(scores), len(toks))
    if L == 0:
        raise SystemExit("No tokens to align.")
    rows = []
    for i in range(L):
        a, b = offs[i]
        seg = text[a:b]
        rows.append(
            {
                "i": i,
                "score": float(scores[i]),
                "char_start": int(a),
                "char_end": int(b),
                "token": toks[i],
                "surface": seg.replace("\n", " "),
            }
        )

    # optional: merge wordpieces (##) into words
    merged = []
    cur = None
    for r in rows:
        if r["token"].startswith("##") and cur is not None:
            cur["score"] += r["score"]  # sum scores across pieces
            cur["char_end"] = r["char_end"]
            cur["token"] += r["token"]
            cur["surface"] = text[cur["char_start"] : cur["char_end"]].replace(
                "\n", " "
            )
        else:
            if cur is not None:
                merged.append(cur)
            cur = dict(r)
    if cur is not None:
        merged.append(cur)

    # write TSV
    import csv

    with open(args.out_tsv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(merged[0].keys()), delimiter="\t")
        w.writeheader()
        for r in merged:
            w.writerow(r)

    print(f"Saved → {args.out_tsv}")


if __name__ == "__main__":
    main()
