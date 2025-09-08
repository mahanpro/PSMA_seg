#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verbose visualization for GPT-RAW windows.

Usage:
python viz_gpt_raw_chunks.py \
  --clean_jsonl /path/to/clean_reports.jsonl \
  --out_dir /path/to/cache_gpt_raw \
  --id 0024 \
  --embed_model text-embedding-3-small \
  --chunk_tokens 16 \
  --stride_tokens 4 \
  --max_preview_chars 140 \
  --reconstruct_if_empty
"""
import argparse, json, re
from pathlib import Path
from typing import List, Optional
import torch
import tiktoken
import numpy as np


def load_clean(clean_jsonl: Path, rid: str) -> Optional[str]:
    found = None
    n = 0
    with clean_jsonl.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            n += 1
            try:
                obj = json.loads(ln)
            except Exception:
                continue
            if str(obj.get("ID") or obj.get("id")) == rid:
                found = (obj.get("clean") or obj.get("Description") or "").strip()
                break
    if not found:
        print(f"[WARN] ID={rid} not found in {clean_jsonl} (scanned {n} lines).")
    return found


def get_encoder(name: str):
    try:
        return tiktoken.encoding_for_model(name)
    except Exception:
        return tiktoken.get_encoding("cl100k_base")


def find_subsequence(hay: List[int], needle: List[int], start_idx: int = 0) -> int:
    n, m = len(hay), len(needle)
    if m == 0:
        return start_idx
    if n < m:
        return -1
    i = max(0, start_idx)
    while i <= n - m:
        if hay[i : i + m] == needle:
            return i
        i += 1
    return -1


def token_windows(text: str, enc, chunk_tokens: int, stride_tokens: int) -> List[str]:
    """Rebuild overlapping token windows from text (like in gpt_raw_embed.py)."""
    toks = enc.encode(text)
    out = []
    i = 0
    while i < len(toks):
        j = min(i + chunk_tokens, len(toks))
        out.append(enc.decode(toks[i:j]))
        if j == len(toks):
            break
        i = max(j - stride_tokens, i + 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean_jsonl", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--id", required=True)
    ap.add_argument("--embed_model", default="text-embedding-3-small")
    ap.add_argument("--chunk_tokens", type=int, default=16)
    ap.add_argument("--stride_tokens", type=int, default=4)
    ap.add_argument("--max_preview_chars", type=int, default=120)
    ap.add_argument("--dump_tsv", default="")
    ap.add_argument(
        "--reconstruct_if_empty",
        action="store_true",
        help="If saved strings are empty, rebuild token windows from raw text so you still get output.",
    )
    args = ap.parse_args()

    rid = args.id
    out_dir = Path(args.out_dir)
    pt_path = out_dir / f"{rid}_gptraw.pt"
    meta_path = out_dir / f"{rid}_gptraw.json"

    print("\n========== GPT-RAW WINDOW VIS (verbose) ==========")
    print(f"ID: {rid}")
    print(f"Paths:\n  PT:   {pt_path}\n  META: {meta_path}\n  CLEAN:{args.clean_jsonl}")

    if not pt_path.exists():
        print(f"[ERROR] Missing PT file: {pt_path}")
        return

    # Load saved data
    try:
        data = torch.load(pt_path, map_location="cpu")
    except Exception as e:
        print(f"[ERROR] Failed to load {pt_path}: {e}")
        return

    strings: List[str] = list(data.get("strings") or [])
    E = data.get("embeddings", None)
    if E is None:
        E = torch.empty(0, 0)
    elif isinstance(E, np.ndarray):
        E = torch.from_numpy(E)
    elif not isinstance(E, torch.Tensor):
        raise TypeError(f"Unexpected embeddings type: {type(E)}")

    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
    else:
        meta = {}

    print(
        f"Loaded: strings={len(strings)}  embeddings_shape={tuple(E.shape)}  dtype={getattr(E,'dtype',None)}"
    )
    if meta:
        print(f"Meta says: {meta}")

    # Load report text
    txt = load_clean(Path(args.clean_jsonl), rid)
    if not txt:
        print(f"[ERROR] Could not load clean text for ID={rid}.")
        return
    enc = get_encoder(args.embed_model)
    full_ids = enc.encode(txt)
    T = len(full_ids)
    print(
        f"Report length: chars={len(txt):,}  tokens≈{T:,} (encoder={args.embed_model})"
    )

    # Reconstruct windows from text if necessary
    if len(strings) == 0 and args.reconstruct_if_empty:
        print(
            "[INFO] Saved strings are empty. Reconstructing from text using chunk/stride…"
        )
        strings = token_windows(txt, enc, args.chunk_tokens, args.stride_tokens)
        print(
            f"[INFO] Reconstructed {len(strings)} windows (no embeddings available for these)."
        )

    if len(strings) == 0:
        print("[WARN] 0 windows. This can happen if:")
        print(
            "  - The embed run produced 0 chunks (e.g., empty report or chunking params filtered everything)"
        )
        print("  - You pointed to the wrong out_dir or wrong ID")
        print(
            "  - The embed job used char-based chunking instead of token-based and never saved strings (unlikely here)"
        )
        return

    # Per-window analysis
    rows = []
    search_from = 0
    norms = []
    for i, s in enumerate(strings):
        s_ids = enc.encode(s)
        start = find_subsequence(
            full_ids, s_ids, start_idx=max(0, search_from - 4 * args.stride_tokens)
        )
        end = (start + len(s_ids) - 1) if start >= 0 else -1
        overlap = 0
        if i > 0 and rows[-1]["end_tok"] >= 0 and start >= 0:
            overlap = max(0, rows[-1]["end_tok"] - start + 1)
        search_from = start if start >= 0 else search_from

        norm = (
            float(torch.linalg.norm(E[i]).item())
            if (E.ndim == 2 and i < E.shape[0])
            else float("nan")
        )
        norms.append(norm)

        preview = re.sub(r"\s+", " ", s)[: args.max_preview_chars]
        rows.append(
            {
                "i": i,
                "chars": len(s),
                "toks": len(s_ids),
                "start_tok": start,
                "end_tok": end,
                "overlap_prev": overlap,
                "emb_norm": norm,
                "preview": preview,
            }
        )

    # Print summary
    finite_norms = [x for x in norms if np.isfinite(x)]
    if finite_norms:
        v = np.array(finite_norms, dtype=np.float64)
        print(
            f"Embedding norms: min={v.min():.3f}  mean={v.mean():.3f}  max={v.max():.3f}"
        )
    else:
        print("Embedding norms: n/a (no embeddings or not aligned with windows)")

    print("\n-- Windows --")
    for r in rows:
        print(
            f"[{r['i']:03d}] chars={r['chars']:4d}  toks={r['toks']:4d}  "
            f"start={r['start_tok']:5d}  end={r['end_tok']:5d}  "
            f"overlap_prev={r['overlap_prev']:3d}  |‖E‖={r['emb_norm']:.3f}  "
            f"| {r['preview']}"
        )

    # Estimated windows for sanity
    est_windows = 1 + max(
        0,
        (T - args.chunk_tokens + (args.chunk_tokens - args.stride_tokens) - 1)
        // max(1, (args.chunk_tokens - args.stride_tokens)),
    )
    print(f"\nEstimated windows (given chunk/stride): ~{est_windows}")

    # Optional TSV dump
    if args.dump_tsv:
        import csv

        with open(args.dump_tsv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"Saved per-window table → {args.dump_tsv}")

    print("==============================================\n")


if __name__ == "__main__":
    main()
