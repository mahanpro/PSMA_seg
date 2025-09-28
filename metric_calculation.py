#!/usr/bin/env python3

from __future__ import annotations
import os
import re
import json
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Tuple, Any
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage as ndi

# ===================
# ===== CONFIG ======
# ===================

# Base project directory (where 'runs' and 'nifti_output_mask_anonymized' live)
BASE_DIR = Path("/home/azureuser/PSMA_seg/PSMA_seg")

# Directory with ground-truth NIfTI masks (files like 0002_MASK.nii.gz, 0121_MASK.nii.gz)
GT_DIR = BASE_DIR / "nifti_output_mask_anonymized"

# Where experiment runs are stored
RUNS_DIR = BASE_DIR / "runs"

# Experiments and their folder prefixes under RUNS_DIR
# Example structure: RUNS_DIR / f"{prefix}_fold{fold}" / f"test_fold{fold}" / "preds" / "{id}_CT.nii.gz"
EXPERIMENTS = [
    ("image", "exp_image_only"),  # image-only baseline uses "exp_image_only"
    ("gpt", "gpt"),
    ("gpt_raw", "gpt_raw"),
    ("msraw", "msraw"),
    ("radgraph", "radgraph"),
]

FOLDS = [0, 1, 2, 3, 4]

# Surface Dice tolerance(s) in millimeters. You can add more like [3.0, 5.0, 10.0]
SURFACE_DICE_TOLS_MM = [5.0]

# Where to save the final aggregated JSON
OUTPUT_JSON = BASE_DIR / "all_experiments_folds_metrics_last.json"

# EXCLUDE_IDS = {"62", "267"}  #### NEW TO EXCLUDE TWO CASES


# =========================
# ====== UTILITIES ========
# =========================


def zfill_id(numeric_id: str, width: int = 4) -> str:
    """Zero-pad numeric IDs like '62' -> '0062'."""
    return str(numeric_id).zfill(width)


def safe_bool_mask(arr) -> np.ndarray:
    """Binarize any array to bool mask (non-zero -> True)."""
    return np.asarray(arr != 0, dtype=bool)


def load_bin_mask(path: Path) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """Load NIfTI and return boolean mask and voxel spacing in mm (sx, sy, sz)."""
    img = nib.load(str(path))
    data = img.get_fdata()
    mask = safe_bool_mask(data)
    spacing = tuple(float(z) for z in img.header.get_zooms()[:3])
    return mask, spacing


def check_same_shape_spacing(pred: np.ndarray, gt: np.ndarray, sp_pred, sp_gt):
    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs gt {gt.shape}")
    # Spacings might differ slightly due to header quirks; prefer GT for reporting.
    return sp_gt


def confusion_counts(pred: np.ndarray, gt: np.ndarray) -> Tuple[int, int, int, int]:
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, np.logical_not(gt)).sum())
    fn = int(np.logical_and(np.logical_not(pred), gt).sum())
    tn = int(np.logical_and(np.logical_not(pred), np.logical_not(gt)).sum())
    return tp, fp, fn, tn


def dice_score(tp, fp, fn) -> float:
    return 2.0 * tp / max(2 * tp + fp + fn, 1e-8)


def iou_score(tp, fp, fn) -> float:
    return tp / max(tp + fp + fn, 1e-8)


def precision_score(tp, fp) -> float:
    return tp / max(tp + fp, 1e-8)


def recall_score(tp, fn) -> float:
    return tp / max(tp + fn, 1e-8)


def specificity_score(tn, fp) -> float:
    return tn / max(tn + fp, 1e-8)


def relative_volume_diff_percent(v_pred: int, v_gt: int) -> float:
    if v_gt == 0:
        return float("nan")
    return (v_pred - v_gt) / v_gt * 100.0


def surface_voxels(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    eroded = ndi.binary_erosion(mask)
    return np.logical_and(mask, np.logical_not(eroded))


def surface_distances_mm(
    pred: np.ndarray, gt: np.ndarray, spacing: Tuple[float, float, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Return distances (mm) from GT surface to PRED and PRED surface to GT respectively."""
    # Handle empties explicitly
    if not pred.any() and not gt.any():
        return np.array([]), np.array([])
    gt_surf = surface_voxels(gt)
    pred_surf = surface_voxels(pred)
    # EDT of complement; sampling=spacing gives distances in mm
    dt_pred = ndi.distance_transform_edt(~pred, sampling=spacing)
    dt_gt = ndi.distance_transform_edt(~gt, sampling=spacing)
    d_gt_to_pred = dt_pred[gt_surf] if gt_surf.any() else np.array([])
    d_pred_to_gt = dt_gt[pred_surf] if pred_surf.any() else np.array([])
    return d_gt_to_pred, d_pred_to_gt


def assd_hd95_mm(
    pred: np.ndarray, gt: np.ndarray, spacing: Tuple[float, float, float]
) -> Tuple[float, float]:
    a, b = surface_distances_mm(pred, gt, spacing)
    if a.size == 0 and b.size == 0:
        return 0.0, 0.0
    all_d = np.concatenate([a, b]) if a.size and b.size else (a if a.size else b)
    assd = float(all_d.mean()) if all_d.size else 0.0
    hd95 = float(np.percentile(all_d, 95)) if all_d.size else 0.0
    return assd, hd95


def surface_dice_at_tolerance(
    pred: np.ndarray, gt: np.ndarray, spacing: Tuple[float, float, float], tau_mm: float
) -> float:
    a, b = surface_distances_mm(pred, gt, spacing)
    la = a.size
    lb = b.size
    if la == 0 and lb == 0:
        return 1.0  # both empty -> perfect
    ok_a = int((a <= tau_mm).sum()) if la else 0
    ok_b = int((b <= tau_mm).sum()) if lb else 0
    denom = max(la + lb, 1e-8)
    return (ok_a + ok_b) / denom


def volume_mm3(mask: np.ndarray, spacing: Tuple[float, float, float]) -> float:
    return float(mask.sum() * np.prod(spacing))


def volume_ml(mask: np.ndarray, spacing: Tuple[float, float, float]) -> float:
    return volume_mm3(mask, spacing) / 1000.0  # 1000 mm^3 = 1 mL


@dataclass
class CaseMetrics:
    id: str
    n_vox_pred: int
    n_vox_gt: int
    vol_pred_ml: float
    vol_gt_ml: float
    dice: float
    iou: float
    precision: float
    recall: float
    specificity: float
    rvd_percent: float
    assd_mm: float
    hd95_mm: float
    sds_mm: Dict[str, float] = field(default_factory=dict)  # e.g. {"5.0": 0.91}


def compute_case_metrics(
    pred_path: Path, gt_path: Path, surface_tols: List[float]
) -> CaseMetrics:
    pred, sp_pred = load_bin_mask(pred_path)
    gt, sp_gt = load_bin_mask(gt_path)

    spacing = check_same_shape_spacing(pred, gt, sp_pred, sp_gt)

    tp, fp, fn, tn = confusion_counts(pred, gt)
    dice = dice_score(tp, fp, fn)
    iou = iou_score(tp, fp, fn)
    prec = precision_score(tp, fp)
    rec = recall_score(tp, fn)
    spec = specificity_score(tn, fp)

    nvp = int(pred.sum())
    nvg = int(gt.sum())

    rvd = relative_volume_diff_percent(nvp, nvg)

    assd, hd95 = assd_hd95_mm(pred, gt, spacing)

    sds = {}
    for tol in surface_tols:
        sds[str(float(tol))] = float(surface_dice_at_tolerance(pred, gt, spacing, tol))

    return CaseMetrics(
        id=pred_path.name.split("_")[0],
        n_vox_pred=nvp,
        n_vox_gt=nvg,
        vol_pred_ml=volume_ml(pred, spacing),
        vol_gt_ml=volume_ml(gt, spacing),
        dice=float(dice),
        iou=float(iou),
        precision=float(prec),
        recall=float(rec),
        specificity=float(spec),
        rvd_percent=float(rvd) if not (np.isnan(rvd)) else None,
        assd_mm=float(assd),
        hd95_mm=float(hd95),
        sds_mm=sds,
    )


def list_ids_for_fold(prefix: str, fold: int) -> List[str]:
    """
    Prefer to read ids from dice_test.json if present (keeps test split exact).
    Fallback: list prediction filenames in preds/ and parse the leading 4-digit ID.
    """
    fold_dir = RUNS_DIR / f"{prefix}_fold{fold}_last" / f"test_fold{fold}"
    dice_json = fold_dir / "dice_test.json"
    if dice_json.exists():
        with open(dice_json, "r") as f:
            data = json.load(f)
        if "ids" in data and isinstance(data["ids"], list):
            return [str(x) for x in data["ids"]]
    # Fallback
    preds_dir = fold_dir / "preds"
    ids = []
    if preds_dir.exists():
        for p in preds_dir.glob("*.nii.gz"):
            m = re.match(r"^(\d{4})_CT\.nii\.gz$", p.name)
            if m:
                ids.append(
                    str(int(m.group(1)))
                )  # store without leading zeros for consistency
    return sorted(ids, key=lambda x: int(x))


def summarize_metrics(
    cases: List[CaseMetrics], surface_tols: List[float]
) -> Dict[str, Any]:
    def safe_stats(vals: List[float]) -> Dict[str, float]:
        arr = np.asarray([v for v in vals if v is not None], dtype=float)
        if arr.size == 0:
            return {
                "mean": None,
                "std": None,
                "median": None,
                "iqr25": None,
                "iqr75": None,
            }
        return {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
            "median": float(np.median(arr)),
            "iqr25": float(np.percentile(arr, 25)),
            "iqr75": float(np.percentile(arr, 75)),
        }

    stats = {}
    # Scalar metrics
    fields = [
        "dice",
        "iou",
        "precision",
        "recall",
        "specificity",
        "rvd_percent",
        "assd_mm",
        "hd95_mm",
        "vol_pred_ml",
        "vol_gt_ml",
    ]
    for f in fields:
        vals = [getattr(c, f) for c in cases]
        stats[f] = safe_stats(vals)

    # Surface Dice tolerances
    sds_stats = {}
    for tol in surface_tols:
        key = str(float(tol))
        vals = [c.sds_mm.get(key, None) for c in cases]
        sds_stats[key] = safe_stats(vals)
    stats["surface_dice_at_tolerance_mm"] = sds_stats

    return stats


def main():
    result: Dict[str, Any] = {
        "dataset": "PSMA test set",
        "ground_truth_dir": str(GT_DIR),
        "n_experiments": len(EXPERIMENTS),
        "folds": FOLDS,
        "surface_dice_tolerances_mm": SURFACE_DICE_TOLS_MM,
        "experiments": {},
    }

    for exp_name, prefix in EXPERIMENTS:
        exp_block: Dict[str, Any] = {"prefix": prefix, "folds": {}}
        all_cases_for_macro: List[CaseMetrics] = []  # concatenate all folds (micro-avg)
        fold_means: Dict[str, List[float]] = {}  # for macro avg over folds

        for fold in FOLDS:
            ids = list_ids_for_fold(prefix, fold)
            # drop unwanted cases (ids are already unpadded like "62")
            # ids = [
            #     i for i in ids if i not in EXCLUDE_IDS
            # ]  #### NEW TO EXCLUDE TWO CASES

            fold_dir = RUNS_DIR / f"{prefix}_fold{fold}_last" / f"test_fold{fold}"
            preds_dir = fold_dir / "preds"

            fold_cases: List[CaseMetrics] = []
            missing: List[str] = []

            for id_str in ids:
                pad = zfill_id(id_str, 4)
                pred_path = preds_dir / f"{pad}_CT.nii.gz"
                gt_path = GT_DIR / f"{pad}_MASK.nii.gz"

                if not pred_path.exists() or not gt_path.exists():
                    missing.append(id_str)
                    continue
                try:
                    cm = compute_case_metrics(pred_path, gt_path, SURFACE_DICE_TOLS_MM)
                    fold_cases.append(cm)
                except Exception as e:
                    missing.append(f"{id_str} (error: {e})")

            # Summaries
            summary = summarize_metrics(fold_cases, SURFACE_DICE_TOLS_MM)

            # Track for overall summaries
            all_cases_for_macro.extend(fold_cases)
            for metric_name, stats in summary.items():
                # stats is dict with mean/std/median... except for surface dice block
                if metric_name == "surface_dice_at_tolerance_mm":
                    for tol_str, tol_stats in stats.items():
                        fold_means.setdefault(f"sds@{tol_str}mm", []).append(
                            tol_stats["mean"]
                            if tol_stats["mean"] is not None
                            else np.nan
                        )
                else:
                    fold_means.setdefault(metric_name, []).append(
                        stats["mean"] if stats["mean"] is not None else np.nan
                    )

            # Write fold block
            exp_block["folds"][f"fold{fold}"] = {
                "n_cases": len(fold_cases),
                "missing": missing,
                "ids": [c.id for c in fold_cases],
                "metrics_per_case": [asdict(c) for c in fold_cases],
                "summary": summary,
            }

        # Experiment-level summaries
        exp_block["overall"] = {
            "micro_summary_over_all_cases": summarize_metrics(
                all_cases_for_macro, SURFACE_DICE_TOLS_MM
            ),
            "macro_mean_of_fold_means": {
                k: float(np.nanmean(v)) if len(v) else None
                for k, v in fold_means.items()
            },
        }

        result["experiments"][exp_name] = exp_block

    # Save
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[OK] Wrote: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
