"""
python psma_models/tools/heat_html.py \
  --id 135 \
  --clean_jsonl ./clean_reports.jsonl \
  --msraw_tsv ./runs/msraw/test_fold0/msraw.tsv \
  --token_scores ./runs/msraw/test_fold0/attn_tokens/135_token_scores.npy \
  --out_html ./runs/msraw/test_fold0/135_msraw_token_heat.html \
  --out_sent_tsv ./runs/msraw/test_fold0/msraw_sentences.tsv \
  --out_sec_tsv ./runs/msraw/test_fold0/msraw_sections.tsv
"""

import argparse, json, html, re
from pathlib import Path
import numpy as np
import pandas as pd

# ---- simple sentence splitter and section header finder ----
SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])|\n+")
SECTION_HDR = re.compile(
    r"(?im)^(Prostatic fossa|Lymph nodes|Skeleton|Viscera)\s*:\s*$"
)


def sentence_spans(text: str):
    bounds = [0]
    for m in SPLIT_PATTERN.finditer(text):
        bounds.append(m.end())
    bounds.append(len(text))
    spans = []
    for i in range(len(bounds) - 1):
        a, b = bounds[i], bounds[i + 1]
        seg = text[a:b]
        a += len(seg) - len(seg.lstrip())
        b -= len(seg) - len(seg.rstrip())
        if a < b:
            spans.append((a, b))
    return spans


def find_sections(text: str):
    spans = []
    pos = 0
    for ln in text.split("\n"):
        st, en = pos, pos + len(ln)
        if SECTION_HDR.match(ln.strip()):
            spans.append((st, en, ln.strip()))
        pos = en + 1
    return spans


def section_for(char_start: int, sec_spans):
    hdr = ""
    for hs, he, h in sec_spans:
        if hs <= char_start:
            hdr = h
        else:
            break
    return hdr


def load_clean_report(clean_jsonl: Path, rid: str) -> str:
    with clean_jsonl.open("r", encoding="utf-8") as f:
        for ln in f:
            j = json.loads(ln)
            if str(j.get("ID") or j.get("id")) == str(f"{int(rid):04d}"):
                return (
                    j.get("clean") or j.get("Description") or j.get("report") or ""
                ).strip()
    raise FileNotFoundError(f"ID {rid} not found in {clean_jsonl}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument(
        "--clean_jsonl",
        required=True,
        help="same JSONL you used when creating MS-RAW offsets (clean text)",
    )
    ap.add_argument(
        "--msraw_tsv",
        required=True,
        help=".../runs/msraw/test_foldk/<ID>_msraw_tokens.tsv",
    )
    ap.add_argument(
        "--token_scores",
        required=True,
        help=".../runs/msraw/test_foldk/attn_tokens/<ID>_token_scores.npy",
    )
    ap.add_argument(
        "--out_html", required=True, help="output HTML with token-level heat coloring"
    )
    ap.add_argument(
        "--out_sent_tsv", required=True, help="output TSV: per-sentence sums/means"
    )
    ap.add_argument("--out_sec_tsv", required=True, help="output TSV: per-section sums")
    args = ap.parse_args()

    rid = str(int(args.id))
    text = load_clean_report(Path(args.clean_jsonl), rid)
    sent_sp = sentence_spans(text)
    sec_sp = find_sections(text)

    # Load TSV (token spans) and scores vector
    df = pd.read_csv(args.msraw_tsv, sep="\t")
    arr = np.load(args.token_scores)

    # Align lengths defensively
    n = min(len(df), len(arr))
    df = df.iloc[:n].copy()
    df["score_arr"] = arr[:n].astype(float)

    # If TSV already has a 'score' column from earlier dumps, prefer the npy
    if "score" in df.columns:
        # Keep both for sanity checks
        diff = np.abs(df["score"].to_numpy(dtype=float) - df["score_arr"].to_numpy())
        n_bad = int((diff > 1e-6).sum())
        if n_bad > 0:
            print(
                f"[warn] {n_bad} token scores differed between TSV and NPY; using NPY."
            )
    df["score"] = df["score_arr"]

    # Build HTML overlay: walk the text and wrap each token span with a colored <span>
    # Normalize scores to [0,1] for alpha
    m = float(df["score"].max()) if len(df) else 0.0
    norm = (df["score"] / m).fillna(0.0).to_numpy() if m > 0 else np.zeros(n, float)

    html_parts = []
    cur = 0
    for idx, row in df.iterrows():
        a = int(row["char_start"])
        b = int(row["char_end"])
        s = float(norm[idx])
        # add untouched text before the token
        if cur < a:
            html_parts.append(html.escape(text[cur:a]))
        # token span
        raw_span = text[a:b]  # use original chars so spacing/punc are exact
        tip = f"tok#{idx} | {row.get('token','')} | score={row['score']:.4f}"
        # red with opacity by score; tweak max alpha to keep it readable
        alpha = min(max(s, 0.0), 1.0) * 0.85
        span_html = (
            f'<span title="{html.escape(tip)}" '
            f'style="background-color: rgba(255,0,0,{alpha:.3f});">'
            f"{html.escape(raw_span)}</span>"
        )
        html_parts.append(span_html)
        cur = b
    # trailing text
    if cur < len(text):
        html_parts.append(html.escape(text[cur:]))

    # Top tokens table (human-readable)
    topk = (
        df.assign(surface=df.get("surface", df.get("token", "")))
        .sort_values("score", ascending=False)
        .head(25)[["i", "surface", "token", "score", "char_start", "char_end"]]
    )

    # Aggregate per sentence
    sent_rows = []
    for si, (sa, sb) in enumerate(sent_sp):
        # tokens that overlap the sentence span
        mask = (df["char_end"] > sa) & (df["char_start"] < sb)
        ss = float(df.loc[mask, "score"].sum())
        sc = int(mask.sum())
        mean = ss / sc if sc > 0 else 0.0
        preview = re.sub(r"\s+", " ", text[sa:sb]).strip()
        sent_rows.append(
            {
                "sent_i": si,
                "char_start": sa,
                "char_end": sb,
                "tok_count": sc,
                "attn_sum": ss,
                "attn_mean": mean,
                "preview": preview,
            }
        )
    df_sent = pd.DataFrame(sent_rows).sort_values("attn_sum", ascending=False)

    # Aggregate per section (by token start position)
    sec_rows = []
    for name in ["Prostatic fossa", "Lymph nodes", "Skeleton", "Viscera", ""]:
        sec_rows.append({"section": name, "attn_sum": 0.0, "tok_count": 0})
    sec_map = {r["section"]: r for r in sec_rows}

    for _, r in df.iterrows():
        sec = section_for(int(r["char_start"]), sec_sp)
        if sec not in sec_map:
            sec_map[sec] = {"section": sec, "attn_sum": 0.0, "tok_count": 0}
        sec_map[sec]["attn_sum"] += float(r["score"])
        sec_map[sec]["tok_count"] += 1
    df_sec = pd.DataFrame(sec_map.values()).sort_values("attn_sum", ascending=False)

    # Write TSVs
    Path(args.out_sent_tsv).parent.mkdir(parents=True, exist_ok=True)
    df_sent.to_csv(args.out_sent_tsv, sep="\t", index=False)
    df_sec.to_csv(args.out_sec_tsv, sep="\t", index=False)

    # Compose HTML
    legend = """
    <div style="margin:8px 0 16px 0;font-size:14px;">
      <b>Legend:</b> red intensity = higher token attention. Hover a token for score.
    </div>"""
    top_table = [
        "<table style='border-collapse:collapse;font-size:13px'>",
        "<tr><th style='text-align:left;padding:4px 8px'>#</th>"
        "<th style='text-align:left;padding:4px 8px'>surface</th>"
        "<th style='text-align:left;padding:4px 8px'>token</th>"
        "<th style='text-align:right;padding:4px 8px'>score</th>"
        "<th style='text-align:right;padding:4px 8px'>start</th>"
        "<th style='text-align:right;padding:4px 8px'>end</th></tr>",
    ]
    for _, r in topk.iterrows():
        top_table.append(
            f"<tr><td style='padding:4px 8px'>{int(r['i'])}</td>"
            f"<td style='padding:4px 8px'>{html.escape(str(r['surface']))}</td>"
            f"<td style='padding:4px 8px;color:#666'>{html.escape(str(r['token']))}</td>"
            f"<td style='padding:4px 8px;text-align:right'>{r['score']:.4f}</td>"
            f"<td style='padding:4px 8px;text-align:right'>{int(r['char_start'])}</td>"
            f"<td style='padding:4px 8px;text-align:right'>{int(r['char_end'])}</td></tr>"
        )
    top_table.append("</table>")

    html_doc = f"""<!doctype html>
<meta charset="utf-8">
<title>MS-RAW token heat — ID {rid}</title>
<style>
body{{font-family:system-ui, sans-serif; line-height:1.5;}}
pre{{white-space:pre-wrap; word-break:break-word;}}
h2{{margin:12px 0 6px 0}}
small{{color:#666}}
</style>
<h2>ID {rid} — MS-RAW token-level attention</h2>
<small>Source: {html.escape(Path(args.msraw_tsv).name)}, {html.escape(Path(args.token_scores).name)}</small>
{legend}
<pre style="padding:12px;border:1px solid #eee;border-radius:8px;background:#fafafa">{''.join(html_parts)}</pre>

<h3 style="margin-top:20px">Top tokens (by score)</h3>
{''.join(top_table)}
"""
    Path(args.out_html).write_text(html_doc, encoding="utf-8")
    print(f"HTML → {args.out_html}")
    print(f"Sentences TSV → {args.out_sent_tsv}")
    print(f"Sections TSV → {args.out_sec_tsv}")


if __name__ == "__main__":
    main()
