#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
detection_compute.py
====================

Heavy pass: loads NIfTI masks, computes lesion-level detection metrics
(C1 any-overlap, C2 IoU>=τ, C3 SUVmax if PET available), plus FPV/FNV (mL)
and lesion counts. Writes:
  - per-case CSVs per fold (fast to re-use later)
  - a compact per-case MEANS CSV across folds (super fast for reports)
  - a JSON summary for completeness

Usage:
# C1+C2 only
python detection_compute.py --outdir /home/azureuser/PSMA_seg/PSMA_seg/detection_outputs

# With PET to enable C3
python detection_compute.py --pet_dir /home/azureuser/PSMA_seg/PSMA_seg/nifti_output_anonymized --pet_pattern "{id}_PT.nii.gz" --outdir /home/azureuser/PSMA_seg/PSMA_seg/detection_outputs


Defaults:
  connectivity = 18
  exclude = {62, 267}
  folds = [0,1,2,3,4]
  experiments map to your runs/ layout
"""

from __future__ import annotations
import argparse, json, math, re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import nibabel as nib
import pandas as pd
from scipy import ndimage as ndi

try:
    import cc3d

    _HAVE_CC3D = True
except Exception:
    _HAVE_CC3D = False

# -------------------- PROJECT DEFAULTS --------------------

BASE_DIR = Path("/home/azureuser/PSMA_seg/PSMA_seg")
GT_DIR = BASE_DIR / "nifti_output_mask_anonymized"
RUNS_DIR = BASE_DIR / "runs"

# (display_name, runs prefix)
EXPERIMENTS = {
    "image": ("Image only(†)", "exp_image_only"),
    "msraw": ("BiomedVLP", "msraw"),
    "radgraph": ("RadGraph", "radgraph"),
    "gpt_raw": ("GPT-raw", "gpt_raw"),
    "gpt": ("GPT-engineered", "gpt"),
}
DEFAULT_ORDER = ["image", "msraw", "radgraph", "gpt_raw", "gpt"]
DEFAULT_FOLDS = [0, 1, 2, 3, 4]
DEFAULT_EXCLUDE = {"62", "267"}  # normalized

# ----------------------------------------------------------


def zfill_id(numeric_id: str, width: int = 4) -> str:
    return str(numeric_id).zfill(width)


def safe_bool_mask(arr) -> np.ndarray:
    return np.asarray(arr != 0, dtype=bool)


def load_bool_nii(path: Path) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    img = nib.load(str(path))
    data = img.get_fdata()
    mask = safe_bool_mask(data)
    spacing = tuple(float(z) for z in img.header.get_zooms()[:3])
    return mask, spacing


def load_float_nii(path: Path) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    img = nib.load(str(path))
    data = img.get_fdata().astype(np.float32, copy=False)
    spacing = tuple(float(z) for z in img.header.get_zooms()[:3])
    return data, spacing


def list_ids_for_fold(prefix: str, fold: int) -> List[str]:
    fold_dir = RUNS_DIR / f"{prefix}_fold{fold}_last" / f"test_fold{fold}"
    dice_json = fold_dir / "dice_test.json"
    if dice_json.exists():
        try:
            data = json.loads(dice_json.read_text())
            if "ids" in data and isinstance(data["ids"], list):
                return [str(x) for x in data["ids"]]
        except Exception:
            pass
    preds_dir = fold_dir / "preds"
    ids = []
    if preds_dir.exists():
        for p in preds_dir.glob("*.nii.gz"):
            m = re.match(r"^(\d{4})_CT\.nii\.gz$", p.name)
            if m:
                ids.append(str(int(m.group(1))))
    return sorted(ids, key=lambda x: int(x))


def check_same_shape(pred: np.ndarray, gt: np.ndarray):
    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs gt {gt.shape}")


def volume_ml(mask: np.ndarray, spacing: Tuple[float, float, float]) -> float:
    return float(mask.sum() * np.prod(spacing) / 1000.0)


def label_components(mask: np.ndarray, connectivity: int) -> Tuple[np.ndarray, int]:
    if _HAVE_CC3D:
        lab, n = cc3d.connected_components(
            mask.astype(np.uint8), connectivity=connectivity, return_N=True
        )
        return lab, int(n)
    if connectivity == 6:
        struct = ndi.generate_binary_structure(rank=3, connectivity=1)
    elif connectivity == 18:
        struct = np.array(
            [
                [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
            ],
            dtype=bool,
        )
    else:
        struct = np.ones((3, 3, 3), dtype=bool)
    lab, n = ndi.label(mask, structure=struct)
    return lab, int(n)


def component_sizes(lab: np.ndarray, n: int) -> np.ndarray:
    return np.bincount(lab.ravel(), minlength=n + 1).astype(np.int64)


def false_positive_volume_ml(gt: np.ndarray, pred: np.ndarray, spacing) -> float:
    lab_p, n_p = label_components(pred, connectivity=18)
    fp_vox = 0
    for i in range(1, n_p + 1):
        comp = lab_p == i
        if not (comp & gt).any():
            fp_vox += int(comp.sum())
    return float(fp_vox * np.prod(spacing) / 1000.0)


def false_negative_volume_ml(gt: np.ndarray, pred: np.ndarray, spacing) -> float:
    lab_g, n_g = label_components(gt, connectivity=18)
    fn_vox = 0
    for j in range(1, n_g + 1):
        comp = lab_g == j
        if not (comp & pred).any():
            fn_vox += int(comp.sum())
    return float(fn_vox * np.prod(spacing) / 1000.0)


def match_max_iou(
    lab_g: np.ndarray, n_g: int, lab_p: np.ndarray, i: int, gt_sizes: np.ndarray
) -> Tuple[Optional[int], float]:
    vox = lab_p == i
    if not vox.any():
        return None, 0.0
    gt_labels = lab_g[vox]
    if gt_labels.size == 0:
        return None, 0.0
    counts = np.bincount(gt_labels, minlength=n_g + 1)
    counts[0] = 0
    j_star = int(np.argmax(counts))
    inter = int(counts[j_star])
    if j_star == 0 or inter == 0:
        return None, 0.0
    pred_size = int(vox.sum())
    gt_size = int(gt_sizes[j_star])
    iou = inter / max(pred_size + gt_size - inter, 1)
    return j_star, float(iou)


def suvmax_in_pred(
    gt_lab: np.ndarray, j: int, pred_mask: np.ndarray, pet: np.ndarray
) -> bool:
    lesion = gt_lab == j
    if not lesion.any():
        return False
    prod = pet * lesion.astype(pet.dtype)
    idx = np.unravel_index(np.argmax(prod), prod.shape)
    return bool(pred_mask[idx])


@dataclass
class DetectionCase:
    id: str
    n_lesions_gt: int
    n_lesions_pred: int
    fpv_ml: float
    fnv_ml: float
    tp_c1: Optional[int] = None
    fp_c1: Optional[int] = None
    fn_c1: Optional[int] = None
    sens_c1: Optional[float] = None
    tp_c2: Optional[int] = None
    fp_c2: Optional[int] = None
    fn_c2: Optional[int] = None
    sens_c2: Optional[float] = None
    tp_c3: Optional[int] = None
    fp_c3: Optional[int] = None
    fn_c3: Optional[int] = None
    sens_c3: Optional[float] = None


def criterion1_counts(lab_g, n_g, lab_p, n_p):
    tp, fp = 0, 0
    for i in range(1, n_p + 1):
        pred_i = lab_p == i
        if np.any(lab_g[pred_i] != 0):
            tp += 1
        else:
            fp += 1
    fn = 0
    for j in range(1, n_g + 1):
        if not np.any(lab_p[lab_g == j] != 0):
            fn += 1
    return tp, fp, fn


def criterion2_counts(lab_g, n_g, lab_p, n_p, iou_thresh: float):
    gt_available = set(range(1, n_g + 1))
    tp, fp = 0, 0
    gt_sizes = component_sizes(lab_g, n_g)
    for i in range(1, n_p + 1):
        j_star, iou = match_max_iou(lab_g, n_g, lab_p, i, gt_sizes)
        if j_star is not None and (iou >= iou_thresh) and (j_star in gt_available):
            tp += 1
            gt_available.remove(j_star)
        else:
            fp += 1
    fn = len(gt_available)
    return tp, fp, fn


def criterion3_counts(lab_g, n_g, lab_p, n_p, pet: np.ndarray):
    gt_available = set(range(1, n_g + 1))
    tp, fp = 0, 0
    gt_sizes = component_sizes(lab_g, n_g)
    for i in range(1, n_p + 1):
        j_star, _ = match_max_iou(lab_g, n_g, lab_p, i, gt_sizes)
        pred_mask = lab_p == i
        if j_star is not None and (j_star in gt_available):
            if suvmax_in_pred(lab_g, j_star, pred_mask, pet):
                tp += 1
                gt_available.remove(j_star)
            else:
                fp += 1
        else:
            fp += 1
    fn = len(gt_available)
    return tp, fp, fn


def compute_detection_for_case(
    gt_path: Path,
    pred_path: Path,
    connectivity: int,
    iou_thresh: float,
    do_c1: bool,
    do_c2: bool,
    do_c3: bool,
    pet_path: Optional[Path],
) -> DetectionCase:
    gt, sp_gt = load_bool_nii(gt_path)
    pred, _ = load_bool_nii(pred_path)
    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs gt {gt.shape}")

    lab_g, n_g = label_components(gt, connectivity)
    lab_p, n_p = label_components(pred, connectivity)

    rec = DetectionCase(
        id=gt_path.name.split("_")[0],
        n_lesions_gt=int(n_g),
        n_lesions_pred=int(n_p),
        fpv_ml=float(false_positive_volume_ml(gt, pred, sp_gt)),
        fnv_ml=float(false_negative_volume_ml(gt, pred, sp_gt)),
    )

    if do_c1:
        tp, fp, fn = criterion1_counts(lab_g, n_g, lab_p, n_p)
        rec.tp_c1, rec.fp_c1, rec.fn_c1 = int(tp), int(fp), int(fn)
        rec.sens_c1 = float(tp / (tp + fn)) if (tp + fn) > 0 else math.nan

    if do_c2:
        tp, fp, fn = criterion2_counts(lab_g, n_g, lab_p, n_p, iou_thresh)
        rec.tp_c2, rec.fp_c2, rec.fn_c2 = int(tp), int(fp), int(fn)
        rec.sens_c2 = float(tp / (tp + fn)) if (tp + fn) > 0 else math.nan

    if do_c3 and pet_path is not None and pet_path.exists():
        pet, _ = load_float_nii(pet_path)
        if pet.shape != gt.shape:
            raise ValueError(
                f"PET shape {pet.shape} != GT shape {gt.shape} for {gt_path.name}"
            )
        tp, fp, fn = criterion3_counts(lab_g, n_g, lab_p, n_p, pet)
        rec.tp_c3, rec.fp_c3, rec.fn_c3 = int(tp), int(fp), int(fn)
        rec.sens_c3 = float(tp / (tp + fn)) if (tp + fn) > 0 else math.nan

    return rec


# ------------------------------ MAIN ------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_dir", type=str, default=str(GT_DIR))
    ap.add_argument("--runs_dir", type=str, default=str(RUNS_DIR))
    ap.add_argument("--outdir", type=str, required=True)

    ap.add_argument(
        "--experiments",
        nargs="+",
        default=DEFAULT_ORDER,
        help="order; names: " + ",".join(EXPERIMENTS.keys()),
    )
    ap.add_argument("--folds", nargs="+", type=int, default=DEFAULT_FOLDS)
    ap.add_argument("--exclude", nargs="*", default=list(DEFAULT_EXCLUDE))

    ap.add_argument("--connectivity", type=int, choices=[6, 18, 26], default=18)
    ap.add_argument("--iou_thresh", type=float, default=0.5)
    ap.add_argument(
        "--criteria", nargs="+", choices=["c1", "c2", "c3"], default=["c1", "c2", "c3"]
    )

    ap.add_argument(
        "--pet_dir",
        type=str,
        default=None,
        help="Directory with PET nifti; required for C3",
    )
    ap.add_argument("--pet_pattern", type=str, default="{id}_PT.nii.gz")

    args = ap.parse_args()

    gt_dir = Path(args.gt_dir)
    runs = Path(args.runs_dir)
    outdir = Path(args.outdir)
    (outdir / "per_case").mkdir(parents=True, exist_ok=True)

    name_to_prefix = {k: EXPERIMENTS[k][1] for k in args.experiments}
    display_order = args.experiments[:]
    exclude_ids = set(args.exclude)

    want_c1 = "c1" in args.criteria
    want_c2 = "c2" in args.criteria
    want_c3 = ("c3" in args.criteria) and (args.pet_dir is not None)
    pet_dir = Path(args.pet_dir) if args.pet_dir else None

    all_results: Dict[str, Any] = {
        "config": {
            "folds": args.folds,
            "experiments": display_order,
            "exclude": sorted(list(exclude_ids)),
            "criteria": args.criteria,
            "connectivity": args.connectivity,
            "iou_thresh": args.iou_thresh,
            "pet_dir": str(pet_dir) if pet_dir else None,
            "pet_pattern": args.pet_pattern,
        },
        "experiments": {},
    }

    # For fast reports: collect per-case metrics per experiment (list over folds)
    per_exp_case_list: Dict[str, Dict[str, List[DetectionCase]]] = {}

    for exp in display_order:
        prefix = name_to_prefix[exp]
        exp_block: Dict[str, Any] = {"prefix": prefix, "folds": {}}
        all_cases: List[DetectionCase] = []

        for fold in args.folds:
            ids = list_ids_for_fold(prefix, fold)
            ids = [
                i
                for i in ids
                if (i not in exclude_ids and i.lstrip("0") not in exclude_ids)
            ]
            preds_dir = (
                runs / f"{prefix}_fold{fold}_last" / f"test_fold{fold}" / "preds"
            )

            fold_cases: List[DetectionCase] = []
            missing: List[str] = []

            for id_str in ids:
                pad = zfill_id(id_str, 4)
                gt_path = gt_dir / f"{pad}_MASK.nii.gz"
                pred_path = preds_dir / f"{pad}_CT.nii.gz"
                if not gt_path.exists() or not pred_path.exists():
                    missing.append(id_str)
                    continue
                pet_path = None
                if want_c3:
                    pp = pet_dir / args.pet_pattern.format(id=pad)
                    pet_path = pp if pp.exists() else None
                try:
                    rec = compute_detection_for_case(
                        gt_path,
                        pred_path,
                        connectivity=args.connectivity,
                        iou_thresh=args.iou_thresh,
                        do_c1=want_c1,
                        do_c2=want_c2,
                        do_c3=want_c3,
                        pet_path=pet_path,
                    )
                    fold_cases.append(rec)
                except Exception as e:
                    missing.append(f"{id_str} (error: {e})")

            # per-fold CSV
            fold_dir = outdir / "per_case" / exp
            fold_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([asdict(c) for c in fold_cases]).to_csv(
                fold_dir / f"fold{fold}_per_case_detection.csv", index=False
            )

            # fold summary (for info)
            def safe_stats(vals):
                a = np.array(
                    [
                        v
                        for v in vals
                        if v is not None
                        and (not isinstance(v, float) or (not math.isnan(v)))
                    ],
                    dtype=float,
                )
                if a.size == 0:
                    return {
                        "mean": None,
                        "std": None,
                        "median": None,
                        "iqr25": None,
                        "iqr75": None,
                    }
                return {
                    "mean": float(a.mean()),
                    "std": float(a.std(ddof=1) if a.size > 1 else 0.0),
                    "median": float(np.median(a)),
                    "iqr25": float(np.percentile(a, 25)),
                    "iqr75": float(np.percentile(a, 75)),
                }

            have_c3_this = want_c3 and any(c.sens_c3 is not None for c in fold_cases)
            fold_summary = {
                "n_cases": len(fold_cases),
                "missing": missing,
                "ids": [c.id for c in fold_cases],
                "sens_c1": safe_stats([c.sens_c1 for c in fold_cases]),
                "sens_c2": safe_stats([c.sens_c2 for c in fold_cases]),
                "fp_c1": safe_stats([c.fp_c1 for c in fold_cases]),
                "fp_c2": safe_stats([c.fp_c2 for c in fold_cases]),
                "fpv_ml": safe_stats([c.fpv_ml for c in fold_cases]),
                "fnv_ml": safe_stats([c.fnv_ml for c in fold_cases]),
            }
            if have_c3_this:
                fold_summary["sens_c3"] = safe_stats([c.sens_c3 for c in fold_cases])
                fold_summary["fp_c3"] = safe_stats([c.fp_c3 for c in fold_cases])

            exp_block["folds"][f"fold{fold}"] = fold_summary
            all_cases.extend(fold_cases)

            # stash for means later
            for c in fold_cases:
                per_exp_case_list.setdefault(exp, {}).setdefault(c.id, []).append(c)

        all_results["experiments"][exp] = exp_block

    # Write JSON config/summaries (for provenance)
    with open(outdir / "detection_summary.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"[OK] Wrote {outdir/'detection_summary.json'}")

    # Build a compact MEANS CSV across folds (super fast to use later)
    rows = []
    metric_keys = [
        "sens_c1",
        "sens_c2",
        "sens_c3",
        "fp_c1",
        "fp_c2",
        "fp_c3",
        "fpv_ml",
        "fnv_ml",
    ]
    for exp, id_map in per_exp_case_list.items():
        for cid, lst in id_map.items():
            row = {"case_id": cid, "experiment": exp}
            for k in metric_keys:
                vals = [
                    getattr(x, k)
                    for x in lst
                    if getattr(x, k) is not None
                    and (
                        not isinstance(getattr(x, k), float)
                        or (not math.isnan(getattr(x, k)))
                    )
                ]
                if len(vals):
                    row[k] = float(np.mean(vals))
            rows.append(row)
    df_means = pd.DataFrame(rows)
    means_csv = outdir / "detection_per_case_means.csv"
    df_means.to_csv(means_csv, index=False)
    print(f"[OK] Wrote {means_csv}")

    # Optional: all raw (by fold) in one CSV for debugging
    raw_rows = []
    for exp, id_map in per_exp_case_list.items():
        for cid, lst in id_map.items():
            for rec in lst:
                d = asdict(rec)
                d["experiment"] = exp
                raw_rows.append(d)
    pd.DataFrame(raw_rows).to_csv(
        outdir / "detection_per_case_allfolds.csv", index=False
    )
    print(f"[OK] Wrote {outdir/'detection_per_case_allfolds.csv'}")


if __name__ == "__main__":
    main()
