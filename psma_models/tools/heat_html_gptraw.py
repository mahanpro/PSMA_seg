"""
python psma_models/tools/heat_html_gptraw.py \
  --id 135 \
  --clean_jsonl ./clean_reports.jsonl \
  --gpt_raw_tsv ./runs/gpt_raw/test_fold0/135_gpt_raw_windows.tsv \
  --token_scores ./runs/gpt_raw/test_fold0/attn_tokens/135_token_scores.npy \
  --out_html ./runs/gpt_raw/test_fold0/135_gptraw_window_heat.html \
  --out_sent_tsv ./runs/gpt_raw/test_fold0/135_gptraw_sentences.tsv \
  --out_sec_tsv ./runs/gpt_raw/test_fold0/135_gptraw_sections.tsv

"""

import argparse, json, html, re
from pathlib import Path
import numpy as np
import pandas as pd

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


def overlap(a1, a2, b1, b2):
    return max(0, min(a2, b2) - max(a1, b1))


def load_clean_report(clean_jsonl: Path, rid: str) -> str:
    with clean_jsonl.open("r", encoding="utf-8") as f:
        for ln in f:
            j = json.loads(ln)
            if str(j.get("ID") or j.get("id")) == f"{int(rid):04d}":
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
        help="same clean JSONL used to make GPT-RAW windows",
    )
    ap.add_argument(
        "--gpt_raw_tsv",
        required=True,
        help=".../runs/gpt_raw/test_foldk/<ID>_gpt_raw_windows.tsv",
    )
    ap.add_argument(
        "--token_scores",
        required=True,
        help=".../runs/gpt_raw/test_foldk/attn_tokens/<ID>_token_scores.npy",
    )
    ap.add_argument("--out_html", required=True)
    ap.add_argument("--out_sent_tsv", required=True)
    ap.add_argument("--out_sec_tsv", required=True)
    args = ap.parse_args()

    rid = str(int(args.id))
    text = load_clean_report(Path(args.clean_jsonl), rid)
    text_len = len(text)
    sent_sp = sentence_spans(text)
    sec_sp = find_sections(text)

    # Load TSV (windows) and NPY (scores)
    df = pd.read_csv(args.gpt_raw_tsv, sep="\t")
    need_cols = {"i", "char_start", "char_end", "string"}
    if not need_cols.issubset(df.columns):
        raise SystemExit(f"TSV missing required columns {need_cols}")
    arr = np.load(args.token_scores).astype(float)

    # Align lengths (common in your dump: NPY has 11, TSV has 10)
    n = min(len(df), arr.size)
    if len(df) != arr.size:
        print(f"[info] length mismatch: TSV={len(df)} NPY={arr.size} → cropping to {n}")
    df = df.iloc[:n].copy().reset_index(drop=True)
    df["score"] = arr[:n]

    # Clip spans to text bounds (defensive)
    df["char_start"] = df["char_start"].clip(lower=0, upper=text_len)
    df["char_end"] = df["char_end"].clip(lower=0, upper=text_len)
    # enforce start <= end
    bad = (df["char_end"] < df["char_start"]).sum()
    if bad:
        raise SystemExit(f"{bad} rows have end < start")

    # Build a non-overlapping segmentation of the text using all boundaries
    boundaries = {0, text_len}
    boundaries.update(df["char_start"].tolist())
    boundaries.update(df["char_end"].tolist())
    cuts = sorted(boundaries)

    # Pre-normalize scores for visualization alpha
    smax = float(df["score"].max()) if len(df) else 0.0

    def norm(s):
        return (s / smax) if smax > 0 else 0.0

    # For each fragment, compute the max window score covering it
    frags = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        if a >= b:
            continue
        # windows covering [a,b)
        cover = df.index[(df["char_start"] < b) & (df["char_end"] > a)]
        if len(cover) == 0:
            s = 0.0
        else:
            s = float(df.loc[cover, "score"].max())
        frags.append((a, b, s))

    # Compose HTML: continuous overlay with max score per fragment
    parts = []
    cur = 0
    for a, b, s in frags:
        if cur < a:
            parts.append(html.escape(text[cur:a]))
        alpha = min(max(norm(s), 0.0), 1.0) * 0.85
        tip = f"max-window score={s:.6f} [{a}:{b}]"
        span = text[a:b]
        parts.append(
            f'<span title="{html.escape(tip)}" '
            f'style="background-color: rgba(0,128,255,{alpha:.3f});">'
            f"{html.escape(span)}</span>"
        )
        cur = b
    if cur < text_len:
        parts.append(html.escape(text[cur:]))

    # Also list windows (order by i) with their own small highlighted snippets
    win_rows = []
    tbl = [
        "<table style='border-collapse:collapse;font-size:13px'>",
        "<tr><th style='text-align:right;padding:4px 8px'>i</th>"
        "<th style='text-align:right;padding:4px 8px'>score</th>"
        "<th style='text-align:right;padding:4px 8px'>start</th>"
        "<th style='text-align:right;padding:4px 8px'>end</th>"
        "<th style='text-align:left;padding:4px 8px'>string</th></tr>",
    ]
    for _, r in df.iterrows():
        si, sj = int(r["char_start"]), int(r["char_end"])
        sc = float(r["score"])
        snippet = text[si:sj]
        alpha = min(max(norm(sc), 0.0), 1.0) * 0.85
        cell = (
            f"<span style='background-color: rgba(0,128,255,{alpha:.3f});'>"
            f"{html.escape(snippet)}</span>"
        )
        tbl.append(
            f"<tr>"
            f"<td style='padding:4px 8px;text-align:right'>{int(r['i'])}</td>"
            f"<td style='padding:4px 8px;text-align:right'>{sc:.6f}</td>"
            f"<td style='padding:4px 8px;text-align:right'>{si}</td>"
            f"<td style='padding:4px 8px;text-align:right'>{sj}</td>"
            f"<td style='padding:4px 8px'>{cell}</td>"
            f"</tr>"
        )
        win_rows.append(
            {"i": int(r["i"]), "score": sc, "char_start": si, "char_end": sj}
        )
    tbl.append("</table>")

    # Per-sentence aggregation (weight by fraction of window covered)
    sent_rows = []
    for si, (sa, sb) in enumerate(sent_sp):
        tot = 0.0
        for _, r in df.iterrows():
            a, b, s = int(r["char_start"]), int(r["char_end"]), float(r["score"])
            ow = overlap(sa, sb, a, b)
            wl = max(b - a, 1)
            if ow > 0:
                tot += s * (ow / wl)
        preview = re.sub(r"\s+", " ", text[sa:sb]).strip()
        sent_rows.append(
            {
                "sent_i": si,
                "char_start": sa,
                "char_end": sb,
                "attn_sum": tot,
                "preview": preview,
            }
        )
    df_sent = pd.DataFrame(sent_rows).sort_values("attn_sum", ascending=False)

    # Per-section aggregation (by window start position, weighted by overlap)
    sec_map = {}  # name -> sum
    for _, r in df.iterrows():
        a, b, s = int(r["char_start"]), int(r["char_end"]), float(r["score"])
        wl = max(b - a, 1)
        # distribute score across sections by overlap fraction
        alloc = {}
        for hs, he, name in sec_sp:
            ow = overlap(a, b, hs, he)
            if ow > 0:
                alloc[name] = alloc.get(name, 0.0) + s * (ow / wl)
        if not alloc:  # no header above → bucket as ""
            name = section_for(a, sec_sp)
            sec_map[name] = sec_map.get(name, 0.0) + s
        else:
            for name, v in alloc.items():
                sec_map[name] = sec_map.get(name, 0.0) + v
    df_sec = pd.DataFrame(
        [{"section": k, "attn_sum": v} for k, v in sec_map.items()]
    ).sort_values("attn_sum", ascending=False)

    # Write TSVs
    Path(args.out_sent_tsv).parent.mkdir(parents=True, exist_ok=True)
    df_sent.to_csv(args.out_sent_tsv, sep="\t", index=False)
    df_sec.to_csv(args.out_sec_tsv, sep="\t", index=False)

    # HTML page
    legend = """
    <div style="margin:8px 0 16px 0;font-size:14px;">
      <b>Legend:</b> blue intensity = higher window attention (max over overlapping windows).
      Hovering shows the max score at that fragment.
    </div>"""
    html_doc = f"""<!doctype html>
<meta charset="utf-8">
<title>GPT-RAW window heat — ID {rid}</title>
<style>
body{{font-family:system-ui, sans-serif; line-height:1.5;}}
pre{{white-space:pre-wrap; word-break:break-word;}}
h2{{margin:12px 0 6px 0}}
small{{color:#666}}
</style>
<h2>ID {rid} — GPT-RAW window attention</h2>
<small>Sources: {html.escape(Path(args.gpt_raw_tsv).name)}, {html.escape(Path(args.token_scores).name)}</small>
{legend}
<pre style="padding:12px;border:1px solid #eee;border-radius:8px;background:#fafafa">{''.join(parts)}</pre>

<h3 style="margin-top:20px">Windows (original order)</h3>
{''.join(tbl)}
"""
    Path(args.out_html).write_text(html_doc, encoding="utf-8")
    print(f"HTML → {args.out_html}")
    print(f"Sentences TSV → {args.out_sent_tsv}")
    print(f"Sections TSV → {args.out_sec_tsv}")


if __name__ == "__main__":
    main()
