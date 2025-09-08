"""
Usage:
python gpt_raw_embed.py \
  --in_clean_jsonl cache/text/clean_reports.jsonl \
  --out_dir cache/text \
  --embed_model text-embedding-3-small \
  --chunk_tokens 16 \
  --stride_tokens 4 \
  --seed 13 \
  --max_chunks 0
"""

# coding: utf-8
import argparse, json, os, re, random, time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import tiktoken

import numpy as np
import torch
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
from openai import APIConnectionError, RateLimitError, APIError

load_dotenv(find_dotenv(), override=True)

# -------------------- Chunking --------------------
SECTION_HDR = re.compile(
    r"(?im)^(Prostatic fossa|Lymph nodes|Skeleton|Viscera)\s*:\s*$"
)
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")


def _normalize_ws(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def chunk_text(
    text: str,
    chunk_chars: int = 2000,
    min_chars: int = 60,
    max_chunks: int = 16,
) -> List[str]:
    """
    Split by section -> sentences -> pack into ~chunk_chars windows.
    Keeps order; drops tiny scraps; hard-caps to max_chunks.
    """
    text = _normalize_ws(text)
    if not text:
        return []

    # Split into section blocks first (if present)
    lines = text.split("\n")
    blocks: List[str] = []
    cur: List[str] = []
    in_section = False
    for ln in lines:
        if SECTION_HDR.match(ln):
            if cur:
                blocks.append("\n".join(cur).strip())
                cur = []
            in_section = True
            cur.append(ln.strip())
        else:
            cur.append(ln.strip())
    if cur:
        blocks.append("\n".join(cur).strip())

    if len(blocks) <= 1 and not in_section:
        # No section headers detected → treat entire text as one block
        blocks = [text]

    # Sentence split per block, then pack
    chunks: List[str] = []
    for b in blocks:
        # keep header with its body
        parts = b.split("\n", 1)
        head = parts[0] if SECTION_HDR.match(parts[0]) else ""
        body = parts[1] if len(parts) > 1 else (parts[0] if not head else "")
        sents = [s.strip() for s in SENT_SPLIT.split(body) if s.strip()]
        pack = head + ("\n" if head and sents else "")
        for s in sents:
            if len(pack) + 1 + len(s) <= chunk_chars:
                pack = (pack + " " + s).strip()
            else:
                if len(pack) >= min_chars:
                    chunks.append(pack)
                pack = (head + "\n" + s).strip() if head else s
                head = ""  # only keep header once
        if len(pack) >= min_chars:
            chunks.append(pack)

    # Fallback if we somehow produced nothing
    if not chunks and len(text) >= min_chars:
        # hard slice the text
        for i in range(0, len(text), chunk_chars):
            ch = text[i : i + chunk_chars].strip()
            if len(ch) >= min_chars:
                chunks.append(ch)

    # cap
    if max_chunks and len(chunks) > max_chunks:
        chunks = chunks[:max_chunks]

    return chunks


# -------------------- OpenAI helpers --------------------
def exponential_backoff(fn, *args, max_tries=6, **kwargs):
    last_exc = None
    for attempt in range(max_tries):
        try:
            return fn(*args, **kwargs)
        except (APIConnectionError, RateLimitError, APIError) as e:
            last_exc = e
            sleep = (2**attempt) + random.random()
            time.sleep(sleep)
    raise RuntimeError(f"OpenAI call failed after {max_tries} tries: {last_exc}")


def embed_strings(client: OpenAI, model: str, strings: List[str]) -> List[List[float]]:
    if not strings:
        return []
    out: List[List[float]] = []
    B = 96
    for i in range(0, len(strings), B):
        batch = strings[i : i + B]
        r = exponential_backoff(client.embeddings.create, model=model, input=batch)
        out.extend([d.embedding for d in r.data])
    return out


# -------------------- IO --------------------
def load_clean_records(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def token_chunks(text: str, model_hint: str, chunk_tokens: int, stride_tokens: int):
    """
    Split `text` into overlapping token windows using tiktoken.
    Returns a list[str], each a decoded window.
    """
    if not text.strip():
        return []
    try:
        enc = tiktoken.encoding_for_model(model_hint)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")  # safe fallback
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
    ap.add_argument("--in_clean_jsonl", required=True)
    ap.add_argument("--out_dir", default="cache/text")
    ap.add_argument("--embed_model", default="text-embedding-3-small")
    ap.add_argument("--chunk_chars", type=int, default=2000)
    ap.add_argument("--min_chars", type=int, default=60)
    ap.add_argument("--max_chunks", type=int, default=0)
    ap.add_argument(
        "--chunk_tokens",
        type=int,
        default=64,
        help="If >0, build chunks by tokens (tiktoken) instead of characters.",
    )
    ap.add_argument(
        "--stride_tokens",
        type=int,
        default=16,
        help="Token overlap between chunks when --chunk_tokens>0.",
    )
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    random.seed(args.seed)
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    inp = Path(args.in_clean_jsonl)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    index_path = out_dir / "index_gpt_raw.jsonl"

    rows = load_clean_records(inp)
    n = 0
    with index_path.open("w", encoding="utf-8") as findex:
        for row in rows:
            rid = str(row.get("ID") or row.get("id") or "")
            clean = (row.get("clean") or row.get("Description") or "").strip()
            if not rid or not clean:
                continue

            if args.chunk_tokens > 0:
                chunks = token_chunks(
                    clean,
                    args.embed_model,  # e.g., "text-embedding-3-small"
                    args.chunk_tokens,
                    args.stride_tokens,
                )
                if args.max_chunks and len(chunks) > args.max_chunks:
                    chunks = chunks[: args.max_chunks]
            else:
                chunks = chunk_text(
                    clean,
                    chunk_chars=args.chunk_chars,
                    min_chars=args.min_chars,
                    max_chunks=args.max_chunks,
                )
            vecs = embed_strings(client, args.embed_model, chunks)
            E = (
                torch.tensor(np.asarray(vecs, dtype="float16"))
                if vecs
                else torch.empty(0, 0, dtype=torch.float16)
            )
            H = int(E.shape[-1]) if E.numel() else 0

            meta_path = out_dir / f"{rid}_gptraw.json"
            pt_path = out_dir / f"{rid}_gptraw.pt"

            meta = {
                "id": rid,
                "embed_model": args.embed_model,
                "num_chunks": len(chunks),
                "hidden_dim": H,
                "chunk_chars": int(args.chunk_chars),
                "max_chunks": int(args.max_chunks),
            }
            with meta_path.open("w", encoding="utf-8") as fjson:
                json.dump(meta, fjson, ensure_ascii=False, indent=2)

            torch.save({"strings": chunks, "embeddings": E}, pt_path)

            findex.write(
                json.dumps(
                    {
                        "id": rid,
                        "clean_json": str(meta_path),
                        "entities_pt": str(pt_path),
                        "hidden_dim": H,
                        "num_entities": len(chunks),
                    }
                )
                + "\n"
            )

            n += 1

    print(f"Processed {n} reports → {out_dir} (index: {index_path})")


if __name__ == "__main__":
    main()
