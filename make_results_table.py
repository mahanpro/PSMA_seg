# /home/azureuser/PSMA_seg/PSMA_seg/make_results_table.py
import pandas as pd
from pathlib import Path

CSV = "/home/azureuser/PSMA_seg/PSMA_seg/stats_vs_image_baseline.csv"
OUT_TEX = Path("/home/azureuser/PSMA_seg/PSMA_seg/stats_outputs/results_table_bold.tex")

DISPLAY_NAME = {
    "image_only": "Image only (baseline)",
    "msraw": "BiomedVLP",
    "radgraph": "RadGraph",
    "gpt_raw": "Gpt-raw",
    "gpt": "Gpt-engineered",
}
MODEL_ORDER = ["image_only", "msraw", "radgraph", "gpt_raw", "gpt"]

METRIC_LABEL = {
    "dice": "Dice",
    "iou": "IoU",
    "precision": "Precision",
    "recall": "Recall",
    "assd_mm": "ASSD (mm)",
    "hd95_mm": "HD95 (mm)",
    "sds@5mm": "SDS@5\\,mm",
}


def mlabel(m):
    return METRIC_LABEL.get(m, m.replace("_", "\\_"))


def higher_better(m):
    return not m.endswith("_mm")


def fmt_mean_sd(mu, sd, is_dist):
    # one math group per cell; safe in Overleaf
    return (
        f"${mu:.1f}\\,\\pm\\,{sd:.1f}$" if is_dist else f"${mu:.3f}\\,\\pm\\,{sd:.3f}$"
    )


def p_with_stars(p):
    if p < 1e-3:
        return f"{p:.3g}\\textsuperscript{{***}}"
    if p < 1e-2:
        return f"{p:.3g}\\textsuperscript{{**}}"
    if p < 0.05:
        return f"{p:.3g}\\textsuperscript{{*}}"
    return f"{p:.3g}"


df = pd.read_csv(CSV)
metrics = df["metric"].unique().tolist()
pref = ["dice", "iou", "precision", "recall", "assd_mm", "hd95_mm", "sds@5mm"]
ordered_metrics = [m for m in pref if m in metrics] + [
    m for m in metrics if m not in pref
]

lines = []
lines += [
    "\\begin{table*}[t]",
    "\\centering",
    "\\caption{Absolute performance and improvements vs Image only. Bold entries indicate \\textbf{significant improvements} (Holm-adjusted one-sided Wilcoxon, $p<0.05$).}",
    "\\label{tab:results}",
    "\\resizebox{\\textwidth}{!}{%",
    "\\begin{tabular}{@{}llcccccc@{}}",
    "\\toprule",
    "\\textbf{Metric} & \\textbf{Model} & \\textbf{Mean$\\pm$SD} & "
    "$\\boldsymbol{\\Delta}$ vs Image only & $\\boldsymbol{r}$ & $\\boldsymbol{P(Y>X)}$ & $\\boldsymbol{p_{\\mathrm{Holm}}}$ \\\\",
    "\\midrule",
]

for m in ordered_metrics:
    sub = df[df["metric"] == m].copy()
    if sub.empty:
        continue
    hb = higher_better(m)
    is_dist = m.endswith("_mm")

    base_mu = float(sub["baseline_mean"].iloc[0])
    base_sd = float(sub["baseline_sd"].iloc[0])
    base_txt = fmt_mean_sd(base_mu, base_sd, is_dist)

    # rows present for this metric (baseline + those present in CSV)
    present_models = ["image_only"] + [
        k for k in MODEL_ORDER[1:] if not sub[sub["model"] == k].empty
    ]
    nrows = len(present_models)

    # ---- Baseline row (with multirow metric cell) ----
    lines.append(
        f"\\multirow{{{nrows}}}{{*}}{{{mlabel(m)}}} & {DISPLAY_NAME['image_only']} & "
        f"{base_txt} & \\textemdash{{}} & -- & -- & -- \\\\"
    )

    # ---- Model rows ----
    for model_key in present_models[1:]:
        r = sub[sub["model"] == model_key].iloc[0]
        mu = float(r["other_mean"])
        sd = float(r["other_sd"])
        p = float(r["p_holm"])
        delta = (mu - base_mu) if hb else (base_mu - mu)  # + = improvement
        arrow = "\\,\\uparrow" if delta > 0 else ("\\,\\downarrow" if delta < 0 else "")
        delta_tex = f"${delta:+.3f}{arrow}$"
        mean_sd_tex = fmt_mean_sd(mu, sd, is_dist)
        rrb = f"{float(r['r_rank_biserial']):+.2f}"
        pys = f"{float(r['prob_superiority']):.2f}"
        ptxt = p_with_stars(p)

        improved_sig = (delta > 0) and (p < 0.05)
        if improved_sig:
            mean_sd_tex = f"\\textbf{{{mean_sd_tex}}}"
            delta_tex = f"\\textbf{{{delta_tex}}}"
            ptxt = f"\\textbf{{{ptxt}}}"

        lines.append(
            f"& {DISPLAY_NAME[model_key]} & {mean_sd_tex} & {delta_tex} & {rrb} & {pys} & {ptxt} \\\\"
        )

    lines.append("\\midrule")

lines += [
    "\\bottomrule",
    "\\end{tabular}%",
    "}",
    "\\vspace{2pt}",
    "\\raggedright\\scriptsize $\\Delta$ is signed so positive indicates improvement (for lower-better metrics the sign is flipped). $r$ = rank-biserial; $P(Y>X)$ = common-language effect size. Stars: * $p<0.05$, ** $p<0.01$, *** $p<0.001$.",
    "\\end{table*}",
]

OUT_TEX.write_text("\n".join(lines))
print(f"[OK] Wrote {OUT_TEX}")
