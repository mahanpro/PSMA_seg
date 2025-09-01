"""
Build a 'MS-RAW' index of token-level embeddings for each report with chunking >512.
Input:  clean_reports.jsonl (from psma_cleaner.py): lines of {"ID": ..., "clean": ...}
Output dir: e.g., cache/text_msraw
  - {ID}_msraw.json         (metadata)
  - {ID}_msraw_tokens.pt    (torch: {"tokens": (L,Ct) float32, "mask": (L,) bool})
  - index_msraw.jsonl       (one-line per ID summary)

python tools/build_msraw_index.py \
  --in_clean_jsonl cache/text/clean_reports.jsonl \
  --out_dir cache/text_msraw \
  --hf_id microsoft/BiomedVLP-CXR-BERT-specialized \
  --max_total_tokens 2048
"""

import argparse, json
from pathlib import Path

import torch
from transformers import BertTokenizerFast, AutoModel


def iter_clean_rows(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                rid = obj.get("ID") or obj.get("id")
                txt = (obj.get("clean") or obj.get("Description") or "").strip()
                if rid and txt:
                    yield rid, txt


def encode_long_text(
    mdl, tok, text: str, max_len: int, stride: int, max_total: int, device: str
):
    """
    Tokenize with overflow to create multiple chunks of length<=max_len (BERT limit),
    run the encoder per-chunk in a single forward (HF supports batched overflow),
    concatenate token embeddings across chunks, then clip to max_total tokens.
    Returns:
      tokens: (L, Ct) float32
      mask:   (L,) bool  (True where valid tokens; here everything up to L is True)
    """
    # build overflow batches
    enc = tok(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_len,
        return_overflowing_tokens=True,
        stride=stride,
    )
    input_keys = {
        k: v.to(device)
        for k, v in enc.items()
        if k in ("input_ids", "attention_mask", "token_type_ids")
    }
    with torch.no_grad():
        hs = mdl(**input_keys).last_hidden_state  # (n_chunks, Lchunk, Ct)

    # IMPORTANT: if you want to drop [CLS]/[SEP], uncomment the slice below
    # chunks = [hs[i, 1:-1, :] for i in range(hs.shape[0])]
    chunks = [hs[i] for i in range(hs.shape[0])]

    tokens = torch.cat(chunks, dim=0)  # (sumL, Ct)
    if max_total > 0 and tokens.shape[0] > max_total:
        tokens = tokens[:max_total, :]

    L, Ct = tokens.shape
    mask = torch.ones(L, dtype=torch.bool, device=tokens.device)
    # move to CPU float32 for storage
    return tokens.detach().float().cpu(), mask.detach().cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in_clean_jsonl", required=True, help="cache/text/clean_reports.jsonl"
    )
    ap.add_argument("--out_dir", default="cache/text_msraw")
    ap.add_argument("--hf_id", default="microsoft/BiomedVLP-CXR-BERT-specialized")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument(
        "--max_len", type=int, default=512, help="per-chunk token limit (model limit)"
    )
    ap.add_argument(
        "--stride", type=int, default=64, help="token overlap between chunks"
    )
    ap.add_argument(
        "--max_total_tokens",
        type=int,
        default=2048,
        help="final cap AFTER concatenation; set 0 to disable",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "msraw_index.jsonl"

    tok = BertTokenizerFast.from_pretrained(args.hf_id, trust_remote_code=True)
    mdl = (
        AutoModel.from_pretrained(args.hf_id, trust_remote_code=True)
        .to(args.device)
        .eval()
    )

    n = 0
    with index_path.open("w", encoding="utf-8") as findex:
        for rid, clean in iter_clean_rows(Path(args.in_clean_jsonl)):
            tokens, mask = encode_long_text(
                mdl,
                tok,
                clean,
                max_len=args.max_len,
                stride=args.stride,
                max_total=args.max_total_tokens,
                device=args.device,
            )
            # save
            meta_path = out_dir / f"{rid}_msraw.json"
            torch_path = out_dir / f"{rid}_msraw_tokens.pt"
            torch.save({"tokens": tokens, "mask": mask}, torch_path)

            meta = {
                "id": rid,
                "hf_id": args.hf_id,
                "hidden_dim": int(tokens.shape[1]) if tokens.ndim == 2 else 0,
                "num_tokens": int(tokens.shape[0]) if tokens.ndim == 2 else 0,
                "clean_jsonl": str(Path(args.in_clean_jsonl).resolve()),
                "entities_pt": str(torch_path),
            }
            with meta_path.open("w", encoding="utf-8") as fjson:
                json.dump(meta, fjson, ensure_ascii=False, indent=2)

            findex.write(
                json.dumps(
                    {
                        "id": rid,
                        "clean_json": str(meta_path),
                        "entities_pt": str(torch_path),
                        "hidden_dim": meta["hidden_dim"],
                        "num_tokens": meta["num_tokens"],
                    }
                )
                + "\n"
            )
            n += 1

    print(f"Processed {n} reports → {out_dir} (index: {index_path})")


if __name__ == "__main__":
    main()
