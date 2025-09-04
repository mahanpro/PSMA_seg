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


def has_lesion_nonzero(mask_path: Path) -> int:
    # Fast & safe check: try nibabel; if missing, treat as 1 (conservative)
    try:
        import nibabel as nib
        import numpy as np

        arr = nib.load(str(mask_path)).get_fdata(caching="unchanged")
        return int(np.any(arr > 0))
    except Exception:
        return 1


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

    # Build row objects and lesion flags
    rows = []
    y = []  # has_lesion (0/1)
    for ID in ids:
        gt_path = Path(gt[ID]).resolve()
        y.append(has_lesion_nonzero(gt_path))
        rows.append(
            {
                "ID": ID,
                "CT": str(Path(ct[ID]).resolve()),
                "PT": str(Path(pet[ID]).resolve()),
                "GT": str(gt_path),
            }
        )

    # Stratified test split + stratified K-fold on the rest
    try:
        from sklearn.model_selection import train_test_split, StratifiedKFold

        test_ids, trainval_ids = set(), []
        idx = list(range(len(rows)))
        tv_idx, te_idx = train_test_split(
            idx, test_size=args.test_frac, stratify=y, random_state=args.seed
        )
        test_ids = {rows[i]["ID"] for i in te_idx}
        trainval_ids = [rows[i]["ID"] for i in tv_idx]

        # stratified folds on trainval
        tv_y = [y[i] for i in tv_idx]
        skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        fold_map = {}
        for k, (_, val_split_idx) in enumerate(skf.split(trainval_ids, tv_y)):
            for ii in val_split_idx:
                fold_map[trainval_ids[ii]] = k
    except Exception:
        # Fallback: your original behavior (random test, modulo folds)
        rnd = random.Random(args.seed)
        rnd.shuffle(ids)
        n_test = max(1, int(round(len(ids) * args.test_frac)))
        test_ids = set(ids[:n_test])
        cv_ids = ids[n_test:]
        fold_map = {ID: (cv_ids.index(ID) % args.folds) for ID in cv_ids}

    # Write CSV
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        fieldnames = ["ID", "CT", "PT", "GT", "split", "fold", "has_lesion"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r, lbl in zip(rows, y):
            ID = r["ID"]
            split = "test" if ID in test_ids else "trainval"
            fold = -1 if split == "test" else int(fold_map.get(ID, 0))
            w.writerow({**r, "split": split, "fold": fold, "has_lesion": lbl})
    print(
        f"Wrote {len(rows)} rows → {args.out_csv} "
        f"(test={sum(1 for r in rows if r['ID'] in test_ids)}, "
        f"trainval={sum(1 for r in rows if r['ID'] not in test_ids)})"
    )


if __name__ == "__main__":
    main()
