import pandas as pd, numpy as np, math
import matplotlib.pyplot as plt
from pathlib import Path

CSV = "/home/azureuser/PSMA_seg/PSMA_seg/stats_vs_image_baseline.csv"
OUTDIR = Path("/home/azureuser/PSMA_seg/PSMA_seg/stats_outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)

# Display names (baseline shown with a dagger; note added under title)
DISPLAY_NAME = {
    "baseline": "Image only(†)",
    "msraw": "BiomedVLP",
    "radgraph": "RadGraph",
    "gpt_raw": "GPT-raw",
    "gpt": "GPT-engineered",
}
MODEL_ORDER = ["baseline", "msraw", "radgraph", "gpt_raw", "gpt"]

# Color-blind–safe palette: single bar color + accent for stars
BAR_COLOR = "#4C78A8"  # blue (Vega)
STAR_COLOR = "#D55E00"  # vermillion (Okabe-Ito)


def higher_better(metric: str) -> bool:
    return not metric.endswith("_mm")


def stars(p):
    return "***" if p < 1e-3 else ("**" if p < 1e-2 else ("*" if p < 0.05 else ""))


def metric_title(m):
    if m == "iou":
        return "IoU"
    return {"assd_mm": "ASSD (mm)", "hd95_mm": "HD95 (mm)", "sds@5mm": "SDS@5 mm"}.get(
        m, m.capitalize()
    )


def order_metrics_avail(metrics):
    pref = ["dice", "iou", "precision", "recall", "assd_mm", "hd95_mm", "sds@5mm"]
    return [m for m in pref if m in metrics] + [m for m in metrics if m not in pref]


df = pd.read_csv(CSV)
metrics = order_metrics_avail(df["metric"].unique().tolist())

# N cases from CSV (same per metric)
try:
    N = int(df["n"].iloc[0])
except Exception:
    N = None

# Grid: 3 columns by default
n_cols = 3 if len(metrics) >= 3 else len(metrics)

# Place SDS@5 mm in the middle of the 3rd row (3×3 layout)
plot_items = metrics[:]
if n_cols == 3 and "sds@5mm" in plot_items:
    idx = plot_items.index("sds@5mm")
    if idx % 3 == 0:  # left slot → insert a placeholder to push to center
        plot_items.insert(idx, "__EMPTY__")

n_m_eff = len(plot_items)
n_rows = math.ceil(n_m_eff / n_cols)

# Bigger, bold fonts everywhere
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

fig, axes = plt.subplots(
    n_rows, n_cols, figsize=(5.6 * n_cols, 4.2 * n_rows), squeeze=False
)

for idx, item in enumerate(plot_items):
    r, c = divmod(idx, n_cols)
    ax = axes[r][c]

    if item == "__EMPTY__":
        ax.axis("off")
        continue

    m = item
    sub = df[df["metric"] == m].copy()
    if sub.empty:
        ax.axis("off")
        continue

    base_mu = float(sub["baseline_mean"].iloc[0])
    base_sd = float(sub["baseline_sd"].iloc[0])

    ys, es, labels, sigmarks = [], [], [], []
    for key in MODEL_ORDER:
        if key == "baseline":
            mu, sd = base_mu, base_sd
            labels.append(DISPLAY_NAME[key])
            ys.append(mu)
            es.append(sd)
            sigmarks.append("")
        else:
            row = sub[sub["model"] == key]
            if row.empty:
                continue
            mu = float(row["other_mean"].iloc[0])
            sd = float(row["other_sd"].iloc[0])
            p = float(row["p_holm"].iloc[0])
            hb = higher_better(m)
            improved = (mu > base_mu) if hb else (mu < base_mu)
            mark = stars(p) if (p < 0.05 and improved) else ""
            labels.append(DISPLAY_NAME[key])
            ys.append(mu)
            es.append(sd)
            sigmarks.append(mark)

    x = np.arange(len(ys))
    bars = ax.bar(
        x,
        ys,
        yerr=es,
        capsize=3,
        color=BAR_COLOR,
        edgecolor="black",
        linewidth=0.8,
        ecolor="black",
    )

    # y-lims & padding so stars sit just above caps
    if ys:
        ymin = min(0.0, min(ys) - 0.05 * max(es + [0.0]))
        ymax = max(y + e for y, e in zip(ys, es))
        yr = ymax - ymin if ymax > ymin else 1.0
        ax.set_ylim(ymin, ymax + 0.12 * yr)
        pad = max(0.014 * yr, 0.02 * (max(es) if len(es) else 1.0))
    else:
        pad = 0.1

    for i, b in enumerate(bars):
        tip = ys[i] + es[i]
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

    # axis labels & aesthetics
    ylab = "Mean ± SD (mm)" if m in ("assd_mm", "hd95_mm") else "Mean ± SD"
    ax.set_ylabel(ylab, fontweight="bold")
    ax.set_xticks(x, labels, rotation=18)
    ax.set_title(metric_title(m), fontweight="bold")
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.25)

    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")

# remove any unused axes
total_slots = n_rows * n_cols
for j in range(n_m_eff, total_slots):
    r, c = divmod(j, n_cols)
    fig.delaxes(axes[r][c])

# Title + compact subtitle (note) at the top to use whitespace
sup = "Absolute performance comparison"
if N is not None:
    sup += f" (N = {N})"
fig.suptitle(sup, y=0.985, fontsize=16, fontweight="bold")

# Put dagger + p-value thresholds under the suptitle
note = "(†) baseline    •    * p<0.05   ** p<0.01   *** p<0.001 (Holm-adjusted one-sided Wilcoxon vs baseline)"
fig.text(
    0.5,
    0.955,
    note,
    ha="center",
    va="top",
    fontsize=14,
    fontweight="bold",
    color="#333333",
)

# Leave extra room at the top for title + note (and reclaim bottom space)
fig.tight_layout(rect=[0, 0.02, 1, 0.92])

out = OUTDIR / "performance_comparison.png"
fig.savefig(out, dpi=300)
print(f"[OK] Saved {out}")
