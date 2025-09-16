"""
python tools/attn_table_gpt_raw.py \
  --id 135 \
  --gpt_raw_cache ./cache_gpt_raw \
  --attn_dir ./runs/gpt_raw/test_fold0/attn_tokens \
  --clean_jsonl ./clean_reports.jsonl \
  --outdir ./runs/gpt_raw/test_fold0

"""

import argparse, json, sys, re
from pathlib import Path
import numpy as np
import pandas as pd


def load_clean_text(clean_jsonl: Path, rid: str) -> str:
    rid = str(int(rid))
    with open(clean_jsonl, "r", encoding="utf-8") as f:
        for ln in f:
            j = json.loads(ln)
            jid = str(j.get("ID") or j.get("id") or "")
            if jid.lstrip("0") == rid.lstrip("0"):
                return (j.get("clean") or j.get("Description") or "").strip()
    raise SystemExit(f"Could not find clean text for ID={rid} in {clean_jsonl}")


def try_load_windows_from_cache(cache_dir: Path, rid: str):
    """
    Expected formats (any one is fine):
      cache/<ID>.json with {"windows":[{"char_start":int,"char_end":int,"text":str}, ...]}
      cache/<ID>.jsonl containing a single line json with that schema
      cache/<ID>.npz with arrays: char_starts, char_ends, texts (object dtype)
    Returns list of dicts: [{char_start, char_end, text}, ...] or None
    """
    cjson = cache_dir / f"{rid}.json"
    cjsonl = cache_dir / f"{rid}.jsonl"
    cnpz = cache_dir / f"{rid}.npz"
    if cjson.exists():
        obj = json.loads(cjson.read_text(encoding="utf-8"))
        return obj.get("windows", None)
    if cjsonl.exists():
        with cjsonl.open("r", encoding="utf-8") as f:
            line = f.readline()
            obj = json.loads(line)
            return obj.get("windows", None)
    if cnpz.exists():
        dat = np.load(cnpz, allow_pickle=True)
        if all(k in dat for k in ("char_starts", "char_ends", "texts")):
            out = []
            for s, e, t in zip(dat["char_starts"], dat["char_ends"], dat["texts"]):
                out.append({"char_start": int(s), "char_end": int(e), "text": str(t)})
            return out
    return None


def naive_sentence_windows(text: str):
    # very light sentence split with indices
    spans = []
    start = 0
    for m in re.finditer(r"[^.!?]+[.!?]|\Z", text, flags=re.S):
        s, e = m.span()
        s = max(s, start)
        if s < e:
            spans.append({"char_start": s, "char_end": e, "text": text[s:e]})
        start = e
    return spans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument(
        "--gpt_raw_cache", required=True, help="Cache built for GPT-RAW windows"
    )
    ap.add_argument(
        "--attn_dir", required=True, help="Directory with <ID>.npy window scores"
    )
    ap.add_argument("--clean_jsonl", default="", help="Optional, only used as fallback")
    ap.add_argument(
        "--outdir", default=".", help="Where to write <ID>_gpt_raw_windows.tsv"
    )
    args = ap.parse_args()

    rid = str(args.id)
    attn_path = Path(args.attn_dir) / f"{rid}_token_scores.npy"
    if not attn_path.exists():
        raise FileNotFoundError(f"Missing scores: {attn_path}")

    scores = np.load(attn_path)  # shape [N_windows]
    cache_dir = Path(args.gpt_raw_cache)
    windows = try_load_windows_from_cache(cache_dir, rid)

    if windows is None:
        if not args.clean_jsonl:
            raise FileNotFoundError(
                f"Couldn't find window metadata in {cache_dir} for {rid}. "
                "Provide --clean_jsonl to fall back to sentence windows."
            )
        text = load_clean_text(Path(args.clean_jsonl), rid)
        windows = naive_sentence_windows(text)

    if len(windows) != len(scores):
        # If off by small margin, clip to min to keep going
        n = min(len(windows), len(scores))
        windows, scores = windows[:n], scores[:n]

    rows = []
    for i, (w, sc) in enumerate(zip(windows, scores)):
        rows.append(
            {
                "i": i,
                "score": float(sc),
                "char_start": int(w["char_start"]),
                "char_end": int(w["char_end"]),
                "string": w.get("text", ""),
            }
        )
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_tsv = outdir / f"{rid}_gpt_raw_windows.tsv"
    pd.DataFrame(rows).to_csv(out_tsv, sep="\t", index=False)
    print(f"Wrote {out_tsv}")


if __name__ == "__main__":
    main()
