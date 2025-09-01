"""
Usage:
python create_manifest.py \
  --pet_dir   PT_DIR \
  --ct_dir    CT_DIR \
  --mask_dir  MASK_DIR \
  --out_csv   data/dataset.csv \
  --folds 5 --test_frac 0.15 --seed 13
"""

import argparse, csv, re, random
from pathlib import Path


def stem_id(p: Path):
    s = p.name.replace(".nii.gz", "")
    return re.sub(r"_(PT|CT|MASK)$", "", s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pet_dir", required=True)
    ap.add_argument("--ct_dir", required=True)
    ap.add_argument("--mask_dir", required=True)
    ap.add_argument("--out_csv", default="data/dataset.csv")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--test_frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    pet = {stem_id(p): p for p in Path(args.pet_dir).glob("*_PT.nii.gz")}
    ct = {stem_id(p): p for p in Path(args.ct_dir).glob("*_CT.nii.gz")}
    gt = {stem_id(p): p for p in Path(args.mask_dir).glob("*_MASK.nii.gz")}
    ids = sorted(set(pet) & set(ct) & set(gt))
    if not ids:
        raise SystemExit("No matching ID triplets found. Check your folders.")

    rnd = random.Random(args.seed)
    rnd.shuffle(ids)
    n_test = max(1, int(round(len(ids) * args.test_frac)))
    test_ids = set(ids[:n_test])
    cv_ids = ids[n_test:]

    rows = []
    # test split: fold set to -1
    for ID in ids:
        split = "test" if ID in test_ids else "trainval"
        fold = -1 if split == "test" else (cv_ids.index(ID) % args.folds)
        rows.append(
            {
                "ID": ID,
                "CT": str(Path(ct[ID]).resolve()),
                "PT": str(Path(pet[ID]).resolve()),
                "GT": str(Path(gt[ID]).resolve()),
                "split": split,
                "fold": fold,
            }
        )

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["ID", "CT", "PT", "GT", "split", "fold"])
        w.writeheader()
        w.writerows(rows)
    print(
        f"Wrote {len(rows)} rows → {args.out_csv} (test={n_test}, trainval={len(cv_ids)})"
    )


if __name__ == "__main__":
    main()
