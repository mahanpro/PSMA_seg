"""
Model-vs-Baseline Statistical Analysis + Plots
==============================================

Reads the aggregated JSON produced by metrics script (metric_calculation.py),
computes paired Wilcoxon signed-rank tests comparing each model to a baseline
(default: "image"), applies Holm-Bonferroni correction across the four comparisons,
computes effect sizes, and saves both a CSV table and figures.

Metrics supported here:
  dice, iou, precision, recall, assd_mm, hd95_mm, sds@5mm

Figures created:
  - Per metric:
      (1) Bar chart: mean ± SD across cases (baseline + all models)
      (2) Violin plot: per-case distributions per model
      (3) Paired lines: baseline vs each non-baseline model (per case)

USAGE EXAMPLE
-------------
python statistical_test.py \
  --json /runs/all_experiments_folds_metrics.json \
  --outdir /stats_outputs \
  --metrics dice iou precision recall assd_mm hd95_mm sds@5mm \
  --baseline image \
  --others msraw radgraph gpt_raw gpt \
  --exclude 62 276 \
  --onesided
"""

import argparse, json, math
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt

# ---------------------------- Helpers ----------------------------


def holm_bonferroni(pvals: List[float]) -> List[float]:
    """Holm-Bonferroni correction (FWER)."""
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
    """Rank-biserial effect size for paired data from Wilcoxon signed-rank."""
    d = y - x
    d = d[d != 0]
    n = d.size
    if n == 0:
        return 0.0
    ranks = np.argsort(np.argsort(np.abs(d))) + 1  # 1..n
    W_pos = ranks[d > 0].sum()
    W_neg = ranks[d < 0].sum()
    return float((W_pos - W_neg) / (n * (n + 1) / 2))


def probability_of_superiority(x: np.ndarray, y: np.ndarray) -> float:
    """Common-language effect size: P(Y > X) ignoring ties."""
    d = y - x
    pos = np.sum(d > 0)
    neg = np.sum(d < 0)
    return float(pos / (pos + neg)) if (pos + neg) > 0 else 0.5


def extract_per_case_mean(
    json_path: Path, metric: str, sds_tol: float = 5.0, exclude_ids: List[str] = None
) -> Dict[str, Dict[str, float]]:
    """
    Returns: {experiment: {id4: mean_across_folds}}
    metric: "dice", "iou", "precision", "recall", "assd_mm", "hd95_mm", or "sds@{tol}mm"
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    exp_blk = data["experiments"]
    out: Dict[str, Dict[str, float]] = {}

    for exp_name, blk in exp_blk.items():
        per_id_vals: Dict[str, List[float]] = {}
        for _, fold_blk in blk["folds"].items():
            for case in fold_blk["metrics_per_case"]:
                cid = case["id"]
                if exclude_ids and (
                    cid.lstrip("0") in exclude_ids or cid in exclude_ids
                ):
                    continue
                if metric.startswith("sds@"):
                    key = str(float(sds_tol))
                    val = case["sds_mm"].get(key, None)
                else:
                    key = metric
                    val = case.get(key, None)
                if val is None or (
                    isinstance(val, float) and (math.isnan(val) or math.isinf(val))
                ):
                    continue
                per_id_vals.setdefault(cid, []).append(val)
        out[exp_name] = {
            cid: float(np.mean(vals))
            for cid, vals in per_id_vals.items()
            if len(vals) > 0
        }
    return out


def collect_case_vectors(
    per_exp: Dict[str, Dict[str, float]], baseline: str, others: List[str]
):
    """Align cases present in all models; return arrays per model and the IDs."""
    ids_common = None
    for e in [baseline] + others:
        ids_e = set(per_exp.get(e, {}).keys())
        ids_common = ids_e if ids_common is None else (ids_common & ids_e)
    ids = sorted(list(ids_common))
    mats = {
        e: np.array([per_exp[e][i] for i in ids], dtype=float)
        for e in [baseline] + others
    }
    return ids, mats


def is_higher_better(metric: str) -> bool:
    return metric not in ("assd_mm", "hd95_mm")


# ---------------------------- Plotting ----------------------------


def bar_mean_sd(
    metric: str, mats: Dict[str, np.ndarray], order: List[str], savepath: Path
):
    means = [float(np.mean(mats[m])) for m in order]
    sds = [float(np.std(mats[m], ddof=1)) for m in order]
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(order))
    ax.bar(x, means, yerr=sds, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_ylabel("Distance (mm)" if metric.endswith("_mm") else "Score")
    ax.set_title(f"{metric}: mean ± SD across cases")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(savepath, dpi=200)
    plt.close(fig)


def violin_distribution(
    metric: str, mats: Dict[str, np.ndarray], order: List[str], savepath: Path
):
    fig, ax = plt.subplots(figsize=(7, 4))
    data = [mats[m] for m in order]
    vp = ax.violinplot(data, showmeans=True, showmedians=False, showextrema=False)
    ax.set_xticks(np.arange(1, len(order) + 1))
    ax.set_xticklabels(order)
    ax.set_ylabel("Distance (mm)" if metric.endswith("_mm") else "Score")
    ax.set_title(f"{metric}: per-case distribution per model")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(savepath, dpi=200)
    plt.close(fig)


def paired_lines(
    metric: str,
    baseline: str,
    mats: Dict[str, np.ndarray],
    model: str,
    ids: List[str],
    savepath: Path,
):
    base = mats[baseline]
    other = mats[model]
    # order by baseline to make lines readable
    order_idx = np.argsort(base)
    base_ord = base[order_idx]
    other_ord = other[order_idx]
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.array([0, 1])
    for b, o in zip(base_ord, other_ord):
        ax.plot(x, [b, o], marker="o", linewidth=1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([baseline, model])
    ax.set_ylabel("Distance (mm)" if metric.endswith("_mm") else "Score")
    ax.set_title(f"{metric}: paired per-case (baseline → {model})")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(savepath, dpi=200)
    plt.close(fig)


# ---------------------------- Main ----------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="Path to aggregated metrics JSON")
    ap.add_argument(
        "--outdir", required=True, help="Directory to write tables and figures"
    )
    ap.add_argument(
        "--metrics",
        nargs="+",
        default=["dice", "iou", "precision", "recall", "assd_mm", "hd95_mm", "sds@5mm"],
    )
    ap.add_argument("--baseline", default="image")
    ap.add_argument(
        "--others", nargs="+", default=["msraw", "radgraph", "gpt_raw", "gpt"]
    )
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument(
        "--onesided",
        action="store_true",
        help="Use one-sided alternative (improvement over baseline)",
    )
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for metric in args.metrics:
        # parse sds tolerance
        sds_tol = 5.0
        if metric.startswith("sds@"):
            try:
                sds_tol = float(metric.split("@")[1].replace("mm", ""))
            except Exception:
                pass

        per_exp = extract_per_case_mean(
            Path(args.json),
            "sds@5mm" if metric.startswith("sds@") else metric,
            sds_tol=sds_tol,
            exclude_ids=args.exclude,
        )
        # Align
        ids, mats = collect_case_vectors(per_exp, args.baseline, args.others)
        model_order = [args.baseline] + args.others

        # Save figures
        bar_mean_sd(metric, mats, model_order, outdir / f"{metric}_bar_mean_sd.png")
        violin_distribution(metric, mats, model_order, outdir / f"{metric}_violin.png")
        for m in args.others:
            paired_lines(
                metric, args.baseline, mats, m, ids, outdir / f"{metric}_paired_{m}.png"
            )

        # Stats vs baseline
        base_vals = mats[args.baseline]
        higher_better = is_higher_better(metric)
        pvals = []
        tmp = []

        for m in args.others:
            other_vals = mats[m]
            x = base_vals.copy()
            y = other_vals.copy()
            if not higher_better:
                x, y = (
                    -x,
                    -y,
                )  # flip so "greater" means improvement for lower-better metrics
            alternative = "greater" if args.onesided else "two-sided"
            try:
                stat = wilcoxon(
                    y - x, zero_method="pratt", alternative=alternative, method="approx"
                )
                p = float(stat.pvalue)
            except Exception:
                p = 1.0
            pvals.append(p)

            diff = other_vals - base_vals
            rb = rank_biserial_from_wilcoxon(base_vals, other_vals)
            ps = probability_of_superiority(base_vals, other_vals)
            tmp.append(
                {
                    "metric": metric,
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

        p_adj = holm_bonferroni(pvals)
        for rec, p_raw, p_h in zip(tmp, pvals, p_adj):
            rec["p_unadjusted"] = p_raw
            rec["p_holm"] = p_h
            all_rows.append(rec)

    # Save table
    df = pd.DataFrame(all_rows)
    csv_out = outdir / "stats_vs_baseline.csv"
    df.to_csv(csv_out, index=False)

    # Markdown summary of significant improvements (Holm < 0.05 in right direction)
    lines = ["# Significant improvements vs baseline (Holm-adjusted p<0.05)", ""]
    for metric in args.metrics:
        sub = df[df["metric"] == metric].copy()
        higher_better = is_higher_better(metric)
        sig_models = []
        for _, r in sub.iterrows():
            diff = float(r["mean_diff"])
            improved = (diff > 0) if higher_better else (diff < 0)
            if improved and float(r["p_holm"]) < 0.05:
                sig_models.append(r["model"])
        mlist = ", ".join(sig_models) if sig_models else "—"
        lines.append(f"- **{metric}**: {mlist}")
    (outdir / "significant_improvements.md").write_text("\n".join(lines))

    print(f"[OK] Wrote: {csv_out}")
    print(f"[OK] Wrote figures to: {outdir}")
    print(f"[OK] Wrote: {outdir / 'significant_improvements.md'}")


if __name__ == "__main__":
    main()
