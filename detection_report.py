#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
detection_report.py
===================

Fast pass: reads MEANS CSV from detection_compute.py and produces:
  - stats_vs_baseline_detection.csv (Wilcoxon + Holm, effect sizes)
  - significant_improvements_detection.md
  - plots/ per-metric bar, violin, paired-lines
  - comparison_grid.png (compact multi-panel figure)

No NIfTI reading here — runs quickly.

Baseline is the first item of --experiments (default: "image").

Usage:
python detection_report.py --means_csv /home/azureuser/PSMA_seg/PSMA_seg/detection_outputs/detection_per_case_means.csv --outdir /home/azureuser/PSMA_seg/PSMA_seg/detection_outputs --experiments image msraw radgraph gpt_raw gpt --onesided
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Display names (match your style)
DISPLAY_NAME = {
    "image": "Image only(†)",
    "msraw": "BiomedVLP",
    "radgraph": "RadGraph",
    "gpt_raw": "GPT-raw",
    "gpt": "GPT-engineered",
}
BAR_COLOR = "#4C78A8"
STAR_COLOR = "#D55E00"


def is_higher_better(metric_key: str) -> bool:
    return metric_key.startswith("sens")  # sensitivities ↑; FPs/FPV/FNV ↓


def stars(p):
    return "***" if p < 1e-3 else ("**" if p < 1e-2 else ("*" if p < 0.05 else ""))


def holm_bonferroni(pvals: List[float]) -> List[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m, dtype=float)
    cmax = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        cmax = max(cmax, val)
        adj[idx] = min(1.0, cmax)
    return adj.tolist()


def rank_biserial_from_wilcoxon(x: np.ndarray, y: np.ndarray) -> float:
    d = y - x
    d = d[d != 0]
    n = d.size
    if n == 0:
        return 0.0
    ranks = np.argsort(np.argsort(np.abs(d))) + 1
    W_pos = ranks[d > 0].sum()
    W_neg = ranks[d < 0].sum()
    return float((W_pos - W_neg) / (n * (n + 1) / 2))


def probability_of_superiority(x: np.ndarray, y: np.ndarray) -> float:
    d = y - x
    pos = np.sum(d > 0)
    neg = np.sum(d < 0)
    return float(pos / (pos + neg)) if (pos + neg) > 0 else 0.5


def metric_nice_title(k: str) -> str:
    # More descriptive, while keeping C1/C2/C3 visible
    mapping = {
        "sens_c1": "Sensitivity (C1; lesion detected)",
        "sens_c2": "Sensitivity (C2; lesion detected with IoU≥τ)",
        "sens_c3": "Sensitivity (C3; lesion detected by SUVmax rule)",
        # False positives per patient: clarify these are lesion *counts*
        "fp_c1": "FP lesions (C1)",
        "fp_c2": "FP lesions (C2)",
        "fp_c3": "FP lesions (C3)",
        # Volumes: clarify per-patient voxel volume
        "fpv_ml": "FP volume (mL; FP voxels)",
        "fnv_ml": "FN volume (mL; FN voxels)",
    }
    return mapping.get(k, k)


def collect_per_metric(
    df: pd.DataFrame, metric_key: str, order: List[str]
) -> Tuple[List[str], Dict[str, np.ndarray]]:
    # rows: case_id, experiment, metrics...
    present = [m for m in order if m in df["experiment"].unique()]
    # intersection of case_ids across all selected experiments with non-NaN metric
    ids = None
    for e in present:
        sub = df[(df["experiment"] == e) & df[metric_key].notna()]
        ids_e = set(sub["case_id"].tolist())
        ids = ids_e if ids is None else (ids & ids_e)
    ids = sorted(list(ids)) if ids else []
    mats = {
        e: df[(df["experiment"] == e) & df["case_id"].isin(ids)][metric_key]
        .astype(float)
        .to_numpy()
        for e in present
    }
    return ids, mats


def bar_mean_sd(
    metric_key: str, order: List[str], mats: Dict[str, np.ndarray], out: Path
):
    plt.rcParams.update(
        {
            "font.size": 14,
            "font.weight": "bold",
            "axes.titlesize": 16,
            "axes.titleweight": "bold",
            "axes.labelsize": 14,
            "axes.labelweight": "bold",
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
        }
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    means, sds, labels, sigmarks = [], [], [], []
    baseline = order[0]
    base = mats[baseline]
    base_mu, base_sd = float(np.mean(base)), float(np.std(base, ddof=1))
    means.append(base_mu)
    sds.append(base_sd)
    labels.append(DISPLAY_NAME[baseline])
    sigmarks.append("")
    hb = is_higher_better(metric_key)
    for m in order[1:]:
        v = mats[m]
        mu, sd = float(np.mean(v)), float(np.std(v, ddof=1))
        means.append(mu)
        sds.append(sd)
        labels.append(DISPLAY_NAME[m])
        x, y = (base, v) if hb else (-base, -v)
        try:
            stat = wilcoxon(
                y - x, zero_method="pratt", alternative="greater", method="approx"
            )
            p = float(stat.pvalue)
            improved = (mu > base_mu) if hb else (mu < base_mu)
            sigmarks.append(stars(p) if (p < 0.05 and improved) else "")
        except Exception:
            sigmarks.append("")
    x = np.arange(len(means))
    bars = ax.bar(
        x,
        means,
        yerr=sds,
        capsize=3,
        color=BAR_COLOR,
        edgecolor="black",
        linewidth=0.8,
        ecolor="black",
    )
    if len(means):
        ymin = min(0.0, min(means) - 0.05 * max(sds + [0.0]))
        ymax = max(m + e for m, e in zip(means, sds))
        yr = ymax - ymin if ymax > ymin else 1.0
        ax.set_ylim(ymin, ymax + 0.12 * yr)
        pad = max(0.014 * yr, 0.02 * (max(sds) if len(sds) else 1.0))
    else:
        pad = 0.1
    for i, b in enumerate(bars):
        tip = means[i] + sds[i]
        if sigmarks[i]:
            ax.text(
                b.get_x() + b.get_width() / 2,
                tip + pad,
                sigmarks[i],
                ha="center",
                va="bottom",
                fontsize=13,
                fontweight="bold",
                color=STAR_COLOR,
            )
    ax.set_xticks(x, labels, rotation=18)
    ax.set_ylabel(
        "Mean ± SD"
        if hb
        else (
            "Mean ± SD (count)" if metric_key.startswith("fp_c") else "Mean ± SD (mL)"
        )
    )
    ax.set_title(metric_nice_title(metric_key))
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=250)
    plt.close(fig)


def violin_plot(
    metric_key: str, order: List[str], mats: Dict[str, np.ndarray], out: Path
):
    fig, ax = plt.subplots(figsize=(7, 4))
    data = [mats[k] for k in order]
    ax.violinplot(data, showmeans=True, showmedians=False, showextrema=False)
    ax.set_xticks(np.arange(1, len(order) + 1))
    ax.set_xticklabels([DISPLAY_NAME[k] for k in order], rotation=18)
    ax.set_ylabel(
        "Score"
        if is_higher_better(metric_key)
        else ("Count" if metric_key.startswith("fp_c") else "mL")
    )
    ax.set_title(metric_nice_title(metric_key))
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(out, dpi=250)
    plt.close(fig)


def paired_lines(
    metric_key: str,
    baseline: str,
    mats: Dict[str, np.ndarray],
    model: str,
    ids: List[str],
    out: Path,
):
    base = mats[baseline]
    other = mats[model]
    order_idx = np.argsort(base)
    base_ord = base[order_idx]
    other_ord = other[order_idx]
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.array([0, 1])
    for b, o in zip(base_ord, other_ord):
        ax.plot(x, [b, o], marker="o", linewidth=1)
    ax.set_xticks([0, 1], [DISPLAY_NAME[baseline], DISPLAY_NAME[model]])
    ax.set_ylabel(
        "Score"
        if is_higher_better(metric_key)
        else ("Count" if metric_key.startswith("fp_c") else "mL")
    )
    ax.set_title(
        f"{metric_nice_title(metric_key)}: paired (baseline → {DISPLAY_NAME[model]})"
    )
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(out, dpi=250)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--means_csv",
        required=True,
        help="Path to detection_per_case_means.csv from detection_compute.py",
    )
    ap.add_argument("--outdir", required=True)
    ap.add_argument(
        "--experiments",
        nargs="+",
        default=["image", "msraw", "radgraph", "gpt_raw", "gpt"],
        help="Order also sets baseline as first item",
    )
    ap.add_argument(
        "--onesided",
        action="store_true",
        help="Use one-sided alternative (improvement over baseline)",
    )
    args = ap.parse_args()

    outdir = Path(args.outdir)
    (outdir / "plots").mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.means_csv)

    # Detect which metrics are available (C3 columns might be missing)
    possible = [
        "sens_c1",
        "sens_c2",
        "sens_c3",
        "fp_c1",
        "fp_c2",
        "fp_c3",
        "fpv_ml",
        "fnv_ml",
    ]
    metrics = [c for c in possible if c in df.columns and df[c].notna().any()]
    order = [e for e in args.experiments if e in df["experiment"].unique()]
    baseline = order[0]
    others = order[1:]

    rows = []
    for mk in metrics:
        ids, mats = collect_per_metric(df, mk, order)
        if not ids:
            continue
        base_vals = mats[baseline]
        pvals = []
        tmp = []
        hb = is_higher_better(mk)
        for m in others:
            other_vals = mats[m]
            x, y = (base_vals, other_vals) if hb else (-base_vals, -other_vals)
            alt = "greater" if args.onesided else "two-sided"
            try:
                stat = wilcoxon(
                    y - x, zero_method="pratt", alternative=alt, method="approx"
                )
                p = float(stat.pvalue)
            except Exception:
                p = 1.0
            diff = other_vals - base_vals
            rb = rank_biserial_from_wilcoxon(base_vals, other_vals)
            ps = probability_of_superiority(base_vals, other_vals)
            tmp.append(
                {
                    "metric": mk,
                    "model": m,
                    "n": len(diff),
                    "baseline_mean": float(base_vals.mean()),
                    "baseline_sd": float(base_vals.std(ddof=1)),
                    "other_mean": float(other_vals.mean()),
                    "other_sd": float(other_vals.std(ddof=1)),
                    "mean_diff": float(diff.mean()),
                    "median_diff": float(np.median(diff)),
                    "r_rank_biserial": float(rb),
                    "prob_superiority": float(ps),
                    "p_unadjusted": None,
                    "p_holm": None,
                }
            )
            pvals.append(p)
        # Holm across models for this metric
        p_adj = holm_bonferroni(pvals)
        for rec, p_raw, p_h in zip(tmp, pvals, p_adj):
            rec["p_unadjusted"] = p_raw
            rec["p_holm"] = p_h
            rows.append(rec)

        # plots
        ids, mats = collect_per_metric(df, mk, order)  # re-collect to ensure matching
        # bar
        bar_mean_sd(mk, order, mats, outdir / f"plots/{mk}_bar_mean_sd.png")
        # violin
        violin_plot(mk, order, mats, outdir / f"plots/{mk}_violin.png")
        # paired lines
        for m in others:
            paired_lines(
                mk, baseline, mats, m, ids, outdir / f"plots/{mk}_paired_{m}.png"
            )

    stats_csv = outdir / "stats_vs_baseline_detection.csv"
    pd.DataFrame(rows).to_csv(stats_csv, index=False)

    # Markdown summary
    lines = ["# Significant improvements vs baseline (Holm-adjusted p<0.05)", ""]
    for mk in metrics:
        sub = pd.DataFrame(rows)
        sub = sub[sub["metric"] == mk].copy()
        hb = is_higher_better(mk)
        sig_models = []
        for _, r in sub.iterrows():
            diff = float(r["mean_diff"])
            improved = (diff > 0) if hb else (diff < 0)
            if improved and float(r["p_holm"]) < 0.05:
                sig_models.append(r["model"])
        mlist = ", ".join(sig_models) if sig_models else "—"
        lines.append(f"- **{metric_nice_title(mk)}**: {mlist}")
    (outdir / "significant_improvements_detection.md").write_text("\n".join(lines))

    # -------------------------------------------------------------------------
    # Compact comparison grid (UPDATED):
    #   Top row:  fp_c1, fp_c2, fp_c3  (FP lesions / patient; with stars)
    #   Bottom:   fpv_ml and fnv_ml centered, adjacent (no stars)
    #   Stars: only *** when Holm-adjusted one-sided p < 1e-3 and improved
    #   Note:  (†) baseline    •    *** p<0.001 (Holm-adjusted one-sided Wilcoxon vs baseline)
    # -------------------------------------------------------------------------
    top_keys = [k for k in ["fp_c1", "fp_c2", "fp_c3"] if k in metrics]
    vol_keys = [k for k in ["fpv_ml", "fnv_ml"] if k in metrics]  # order: FPV, FNV

    if len(top_keys) or len(vol_keys):
        plt.rcParams.update(
            {
                "font.size": 14,
                "font.weight": "bold",
                "axes.titlesize": 16,
                "axes.titleweight": "bold",
                "axes.labelsize": 14,
                "axes.labelweight": "bold",
                "xtick.labelsize": 13,
                "ytick.labelsize": 13,
            }
        )

        # Use a 2 x 6 GridSpec:
        #   Top row: columns [0:2], [2:4], [4:6]
        #   Bottom row (centered, adjacent): [1:3], [3:5]
        fig = plt.figure(figsize=(18, 9))
        gs = GridSpec(2, 6, figure=fig)

        top_axes = [
            fig.add_subplot(gs[0, 0:2]),
            fig.add_subplot(gs[0, 2:4]),
            fig.add_subplot(gs[0, 4:6]),
        ]
        bottom_axes = [
            fig.add_subplot(gs[1, 1:3]),
            fig.add_subplot(gs[1, 3:5]),
        ]

        def grid_panel(ax, mk: str, show_stars: bool) -> int:
            ids, mats = collect_per_metric(df, mk, order)
            if not ids:
                ax.axis("off")
                return 0

            baseline_key = order[0]
            base = mats[baseline_key]
            base_mu, base_sd = float(np.mean(base)), float(np.std(base, ddof=1))
            means = [base_mu]
            sds = [base_sd]
            labels = [DISPLAY_NAME[baseline_key]]

            hb = is_higher_better(mk)
            base_t = base if hb else -base

            p_raw = []
            improved_flags = []

            for k in order[1:]:
                v = mats[k]
                mu = float(np.mean(v))
                sd = float(np.std(v, ddof=1))
                means.append(mu)
                sds.append(sd)
                labels.append(DISPLAY_NAME[k])

                if show_stars:
                    v_t = v if hb else -v
                    try:
                        stat = wilcoxon(
                            v_t - base_t,
                            zero_method="pratt",
                            alternative="greater",
                            method="approx",
                        )
                        p = float(stat.pvalue)
                    except Exception:
                        p = 1.0
                    p_raw.append(p)
                    improved_flags.append((mu > base_mu) if hb else (mu < base_mu))

            # Holm adjust within panel (if we plan to show stars)
            p_holm = holm_bonferroni(p_raw) if (show_stars and p_raw) else []

            # Bars
            x = np.arange(len(means))
            bars = ax.bar(
                x,
                means,
                yerr=sds,
                capsize=3,
                color=BAR_COLOR,
                edgecolor="black",
                linewidth=0.8,
                ecolor="black",
            )

            if len(means):
                # keep 0 lower bound for count/volume panels
                ymin_candidate = min(means) - 0.05 * max(sds + [0.0])
                ymin = 0.0 if not is_higher_better(mk) else min(0.0, ymin_candidate)
                ymax = max(m + e for m, e in zip(means, sds))
                yr = ymax - ymin if ymax > ymin else 1.0
                ax.set_ylim(ymin, ymax + 0.12 * yr)
                pad = max(0.014 * yr, 0.02 * (max(sds) if len(sds) else 1.0))
            else:
                pad = 0.1

            # Stars: ONLY for non-volume panels
            if show_stars:
                for i, (p_adj, imp) in enumerate(zip(p_holm, improved_flags), start=1):
                    if imp and p_adj < 1e-3:
                        b = bars[i]
                        tip = means[i] + sds[i]
                        ax.text(
                            b.get_x() + b.get_width() / 2,
                            tip + pad,
                            "***",
                            ha="center",
                            va="bottom",
                            fontsize=13,
                            fontweight="bold",
                            color=STAR_COLOR,
                        )

            # Labels
            ax.set_xticks(x, labels, rotation=18)
            ax.set_ylabel(
                "Mean ± SD (count)" if mk.startswith("fp_c") else "Mean ± SD (mL)"
            )
            ax.set_title(metric_nice_title(mk))
            ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.25)
            return len(ids)

        # ----- draw top row (with stars) -----
        Ns = []
        for col in range(3):
            if col < len(top_keys):
                Ns.append(grid_panel(top_axes[col], top_keys[col], show_stars=True))
            else:
                top_axes[col].axis("off")

        # ----- draw bottom row (centered, adjacent; NO stars) -----
        Nb = []
        for i in range(2):
            if i < len(vol_keys):
                Nb.append(grid_panel(bottom_axes[i], vol_keys[i], show_stars=False))
            else:
                bottom_axes[i].axis("off")

        # Choose N from first available panel
        N_all = next((n for n in Ns + Nb if n > 0), 0)

        fig.suptitle(
            f"Lesion-level detection errors across models (N = {N_all})",
            y=0.985,
            fontsize=16,
            fontweight="bold",
        )
        note = "(†) baseline    •    *** p<0.001 (Holm-adjusted one-sided Wilcoxon vs baseline)"
        fig.text(
            0.5,
            0.955,
            note,
            ha="center",
            va="top",
            fontsize=13,
            fontweight="bold",
            color="#333333",
        )

        fig.tight_layout(rect=[0, 0.02, 1, 0.92])
        fig.savefig(outdir / "comparison_grid.png", dpi=300)
        plt.close(fig)

    print(f"[OK] Wrote {stats_csv}")
    print(f"[OK] Wrote plots to {outdir/'plots'}")
    print(f"[OK] Wrote {outdir/'significant_improvements_detection.md'}")
    if (outdir / "comparison_grid.png").exists():
        print(f"[OK] Wrote {outdir/'comparison_grid.png'}")


if __name__ == "__main__":
    main()
