"""
python psma_models/tools/attn_table_gpt_engineered.py \
  --id 135 \
  --gpt_root ./cache_gpt \
  --attn_dir ./runs/gpt/test_fold0/attn_tokens \
  --out_tsv ./runs/gpt/test_fold0/gpt_engineered.tsv

"""

import argparse, json
from pathlib import Path
import numpy as np
import torch


def load_index(index_jsonl: Path) -> dict:
    lut = {}
    with open(index_jsonl, "r", encoding="utf-8") as f:
        for ln in f:
            if ln.strip():
                j = json.loads(ln)
                rid = str(j.get("id"))
                lut[rid] = j
    return lut


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument(
        "--gpt_root", required=True, help="dir with index_gpt.jsonl & *_gpt_entities.pt"
    )
    ap.add_argument("--attn_dir", required=True, help="runs/.../attn_tokens")
    ap.add_argument("--out_tsv", required=True)
    args = ap.parse_args()

    rid = str(int(args.id))
    rid_z = f"{int(args.id):04d}"

    idx = load_index(Path(args.gpt_root) / "index_gpt.jsonl")
    rec = idx.get(rid_z) or idx.get(rid)
    if not rec:
        raise SystemExit(f"missing index row for id={rid}")

    pt_path = Path(rec["entities_pt"])
    if not pt_path.exists():
        raise SystemExit(f"missing entities_pt: {pt_path}")
    pack = torch.load(pt_path, map_location="cpu")
    strings = list(pack.get("strings") or [])

    scores = np.load(Path(args.attn_dir) / f"{rid}_token_scores.npy")
    L = min(len(strings), len(scores))
    rows = [
        {"i": i, "score": float(scores[i]), "string": strings[i][:200]}
        for i in range(L)
    ]
    rows.sort(key=lambda r: r["score"], reverse=True)

    import csv

    with open(args.out_tsv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"Saved → {args.out_tsv}")


if __name__ == "__main__":
    main()
