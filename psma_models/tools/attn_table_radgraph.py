"""
python psma_models/tools/attn_table_radgraph.py \
  --id 135 \
  --radgraph_root ./cache_radgraph \
  --attn_dir ./runs/radgraph/test_fold0/attn_tokens \
  --out_tsv ./runs/radgraph/test_fold0/radgraph.tsv

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
        "--radgraph_root",
        required=True,
        help="dir with index.jsonl & {id}.json / {id}_entities.pt",
    )
    ap.add_argument("--attn_dir", required=True, help="runs/.../attn_tokens")
    ap.add_argument("--out_tsv", required=True)
    args = ap.parse_args()

    rid = str(int(args.id))
    rid_z = f"{int(args.id):04d}"

    idx = load_index(Path(args.radgraph_root) / "index.jsonl")
    rec = idx.get(rid_z) or idx.get(rid)
    if not rec:
        raise SystemExit(f"missing index row for id={rid}")

    meta = json.load(open(rec["clean_json"], "r", encoding="utf-8"))
    ent_pt = Path(rec["entities_pt"])
    if not ent_pt.exists():
        raise SystemExit(f"missing entities_pt: {ent_pt}")

    pack = torch.load(ent_pt, map_location="cpu")
    eids = list(pack.get("eids") or [])
    entity_meta = meta.get("entity_meta", {})

    scores_npy = Path(args.attn_dir) / f"{rid}_token_scores.npy"
    if not scores_npy.exists():
        raise SystemExit(f"missing scores: {scores_npy}")
    scores = np.load(scores_npy)

    L = min(len(eids), len(scores))
    rows = []
    for i in range(L):
        eid = eids[i]
        em = entity_meta.get(eid, {})
        rows.append(
            {
                "i": i,
                "score": float(scores[i]),
                "label": str(em.get("label", "")),
                "tokens": str(em.get("tokens", ""))[:200],
                "start_ix": int(em.get("start_ix", -1)),
                "end_ix": int(em.get("end_ix", -1)),
            }
        )
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
