"""
python psma_models/tools/make_radgraph_heat_html.py \
  --id 135 \
  --clean_jsonl ./clean_reports.jsonl \
  --radgraph_tsv ./runs/radgraph/test_fold0/radgraph.tsv \
  --token_scores ./runs/radgraph/test_fold0/attn_tokens/135_token_scores.npy \
  --out_html ./runs/radgraph/test_fold0/135_radgraph_heat.html \
  --out_sent_tsv ./runs/radgraph/test_fold0/135_radgraph_sentences.tsv \
  --out_sec_tsv ./runs/radgraph/test_fold0/135_radgraph_sections.tsv \
  --out_label_tsv ./runs/radgraph/test_fold0/135_radgraph_labels.tsv \
  --out_unmatched_tsv ./runs/radgraph/test_fold0/135_radgraph_unmatched.tsv

"""

import argparse, json, html, re
from pathlib import Path
import numpy as np
import pandas as pd

SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])|\n+")
SECTION_HDR = re.compile(
    r"(?im)^(Prostatic fossa|Lymph nodes|Skeleton|Viscera)\s*:\s*$"
)


# ----------------- helpers -----------------
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


# Normalize text to improve fuzzy matching and also keep char index mapping
def normalize_with_map(s: str):
    # unify special characters
    s = s.replace("\u00d7", "x")  # × -> x
    s = s.replace("\u2013", "-").replace("\u2014", "-")  # en/em dash -> hyphen
    s = (
        s.replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
    )
    out = []
    map_idx = []
    prev_space = False
    for i, ch in enumerate(s):
        if ch.isspace():
            if not prev_space:
                out.append(" ")
                map_idx.append(i)
            prev_space = True
            continue
        prev_space = False
        if ch.isalnum() or ch in ".,:=x/()+-":
            out.append(ch.lower())
            map_idx.append(i)
        # else: drop
    # collapse spaces again
    norm, norm_map = [], []
    last_space = False
    for ch, idx in zip(out, map_idx):
        if ch == " ":
            if last_space:
                continue
            last_space = True
        else:
            last_space = False
        norm.append(ch)
        norm_map.append(idx)
    return "".join(norm), norm_map


# greedy fuzzy find: exact on normalized; else longest 8→3 token slice
def find_span_in_text(report_text: str, query: str):
    if not query:
        return None
    norm_text, tmap = normalize_with_map(report_text)
    norm_q, _ = normalize_with_map(query)

    pos = norm_text.find(norm_q)
    if pos >= 0:
        a = tmap[pos]
        b = tmap[min(pos + len(norm_q) - 1, len(tmap) - 1)] + 1
        return (a, b)

    toks = [w for w in re.split(r"\s+", norm_q) if w]
    if not toks:
        return None
    for L in range(min(12, len(toks)), max(3, min(12, len(toks))) - 1, -1):
        for i in range(0, len(toks) - L + 1):
            sub = " ".join(toks[i : i + L])
            pos = norm_text.find(sub)
            if pos >= 0:
                a = tmap[pos]
                b = tmap[min(pos + len(sub) - 1, len(tmap) - 1)] + 1
                return (a, b)
    return None


# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument(
        "--clean_jsonl",
        required=True,
        help="clean reports JSONL (same canonical text used elsewhere)",
    )
    ap.add_argument(
        "--radgraph_tsv",
        required=True,
        help=".../runs/radgraph/test_foldk/<ID>_radgraph_entities.tsv",
    )
    ap.add_argument(
        "--token_scores",
        required=True,
        help=".../runs/radgraph/test_foldk/attn_tokens/<ID>_token_scores.npy",
    )
    ap.add_argument("--out_html", required=True)
    ap.add_argument("--out_sent_tsv", required=True)
    ap.add_argument("--out_sec_tsv", required=True)
    ap.add_argument("--out_label_tsv", required=True)
    ap.add_argument("--out_unmatched_tsv", required=True)
    ap.add_argument(
        "--min_chars",
        type=int,
        default=2,
        help="skip matches whose cleaned evidence length < min_chars",
    )
    args = ap.parse_args()

    rid = str(int(args.id))
    text = load_clean_report(Path(args.clean_jsonl), rid)
    text_len = len(text)
    sent_sp = sentence_spans(text)
    sec_sp = find_sections(text)

    # TSV + NPY
    df = pd.read_csv(args.radgraph_tsv, sep="\t")
    if not {"i", "tokens"}.issubset(df.columns):
        raise SystemExit(
            "TSV must have columns at least: i, tokens (label, start_ix, end_ix optional)."
        )
    arr = np.load(args.token_scores).astype(float)

    # align/crop
    n = min(len(df), arr.size)
    if len(df) != arr.size:
        print(f"[info] length mismatch: TSV={len(df)} NPY={arr.size} → cropping to {n}")
    df = df.iloc[:n].copy().reset_index(drop=True)
    df["score"] = arr[:n]

    # evidence string: use the 'tokens' cell as the snippet
    evidences = df["tokens"].astype(str).tolist()

    # map evidence -> char spans
    starts, ends, matched = [], [], []
    for ev in evidences:
        ev_clean = re.sub(r"\s+", " ", ev.replace("×", "x")).strip()
        # discard tiny things if requested (e.g., "cm", "x")
        keep = len(re.sub(r"[^A-Za-z0-9]", "", ev_clean)) >= args.min_chars
        if not keep:
            starts.append(-1)
            ends.append(-1)
            matched.append(False)
            continue
        span = find_span_in_text(text, ev_clean)
        if span is None:
            starts.append(-1)
            ends.append(-1)
            matched.append(False)
        else:
            a, b = span
            starts.append(max(0, a))
            ends.append(min(text_len, b))
            matched.append(True)
    df["evidence"] = evidences
    df["char_start"] = starts
    df["char_end"] = ends
    df["matched"] = matched

    # ---- HTML overlay (max over overlapping evidence spans) ----
    smax = float(df.loc[df["matched"], "score"].max()) if df["matched"].any() else 0.0

    def norm(s):
        return (s / smax) if smax > 0 else 0.0

    boundaries = {0, text_len}
    for a, b, m in zip(df["char_start"], df["char_end"], df["matched"]):
        if m:
            boundaries.add(int(a))
            boundaries.add(int(b))
    cuts = sorted(boundaries)

    frags = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        if a >= b:
            continue
        cover = df.index[
            (df["matched"]) & (df["char_start"] < b) & (df["char_end"] > a)
        ]
        if len(cover) == 0:
            s = 0.0
        else:
            s = float(df.loc[cover, "score"].max())
        frags.append((a, b, s))

    parts, cur = [], 0
    for a, b, s in frags:
        if cur < a:
            parts.append(html.escape(text[cur:a]))
        alpha = min(max(norm(s), 0.0), 1.0) * 0.85
        tip = f"radgraph score={s:.6f} [{a}:{b}]"
        parts.append(
            f'<span title="{html.escape(tip)}" '
            f'style="background-color: rgba(0,160,120,{alpha:.3f});">'
            f"{html.escape(text[a:b])}</span>"
        )
        cur = b
    if cur < text_len:
        parts.append(html.escape(text[cur:]))

    # ---- Per-sentence aggregation (weighted by overlap fraction) ----
    sent_rows = []
    for si, (sa, sb) in enumerate(sent_sp):
        tot = 0.0
        for _, r in df[df["matched"]].iterrows():
            a, b, sc = int(r["char_start"]), int(r["char_end"]), float(r["score"])
            ow = overlap(sa, sb, a, b)
            if ow > 0:
                tot += sc * (ow / max(b - a, 1))
        prev = re.sub(r"\s+", " ", text[sa:sb]).strip()
        sent_rows.append(
            {
                "sent_i": si,
                "char_start": sa,
                "char_end": sb,
                "attn_sum": tot,
                "preview": prev,
            }
        )
    df_sent = pd.DataFrame(sent_rows).sort_values("attn_sum", ascending=False)

    # ---- Per-section aggregation (distribute by overlap) ----
    sec_map = {}
    for _, r in df[df["matched"]].iterrows():
        a, b, sc = int(r["char_start"]), int(r["char_end"]), float(r["score"])
        wl = max(b - a, 1)
        alloc_any = False
        for hs, he, name in sec_sp:
            ow = overlap(a, b, hs, he)
            if ow > 0:
                sec_map[name] = sec_map.get(name, 0.0) + sc * (ow / wl)
                alloc_any = True
        if not alloc_any:
            name = section_for(a, sec_sp)
            sec_map[name] = sec_map.get(name, 0.0) + sc
    df_sec = pd.DataFrame(
        [{"section": k, "attn_sum": v} for k, v in sec_map.items()]
    ).sort_values("attn_sum", ascending=False)

    # ---- Per-label aggregation ----
    lab_map = {}
    for _, r in df[df["matched"]].iterrows():
        lab = str(r.get("label", "") or "")
        lab_map[lab] = lab_map.get(lab, 0.0) + float(r["score"])
    df_lab = pd.DataFrame(
        [{"label": k, "attn_sum": v} for k, v in lab_map.items()]
    ).sort_values("attn_sum", ascending=False)

    # ---- Unmatched rows (for debugging) ----
    df_unmatched = df[~df["matched"]][["i", "score", "label", "tokens"]].copy()

    # ---- Items table ----
    tbl = [
        "<table style='border-collapse:collapse;font-size:13px'>",
        "<tr><th style='text-align:right;padding:4px 8px'>i</th>"
        "<th style='text-align:right;padding:4px 8px'>score</th>"
        "<th style='text-align:left;padding:4px 8px'>label</th>"
        "<th style='text-align:left;padding:4px 8px'>tokens</th>"
        "<th style='text-align:right;padding:4px 8px'>start</th>"
        "<th style='text-align:right;padding:4px 8px'>end</th></tr>",
    ]
    for _, r in df.iterrows():
        sc = float(r["score"])
        a = int(r["char_start"])
        b = int(r["char_end"])
        alpha = (
            min(max((sc / smax) if smax > 0 else 0.0, 0.0), 1.0) * 0.85
            if r["matched"]
            else 0.0
        )
        tok_cell = html.escape(str(r["tokens"]))
        if r["matched"]:
            tok_cell = f"<span style='background-color: rgba(0,160,120,{alpha:.3f});'>{tok_cell}</span>"
        else:
            tok_cell = f"<span style='opacity:.6'>{tok_cell} (unmatched)</span>"
        tbl.append(
            f"<tr><td style='padding:4px 8px;text-align:right'>{int(r['i'])}</td>"
            f"<td style='padding:4px 8px;text-align:right'>{sc:.6f}</td>"
            f"<td style='padding:4px 8px'>{html.escape(str(r.get('label','')))}</td>"
            f"<td style='padding:4px 8px'>{tok_cell}</td>"
            f"<td style='padding:4px 8px;text-align:right'>{a if r['matched'] else '-'}</td>"
            f"<td style='padding:4px 8px;text-align:right'>{b if r['matched'] else '-'}</td></tr>"
        )
    tbl.append("</table>")

    legend = """
    <div style="margin:8px 0 16px 0;font-size:14px;">
      <b>Legend:</b> green intensity = higher RadGraph attention (max over overlapping entity spans).
      Items that couldn’t be aligned are listed as <i>unmatched</i>.
    </div>"""

    html_doc = f"""<!doctype html>
<meta charset="utf-8">
<title>RadGraph heat — ID {rid}</title>
<style>
body{{font-family:system-ui, sans-serif; line-height:1.5;}}
pre{{white-space:pre-wrap; word-break:break-word;}}
h2{{margin:12px 0 6px 0}}
small{{color:#666}}
</style>
<h2>ID {rid} — RadGraph attention</h2>
<small>Sources: {html.escape(Path(args.radgraph_tsv).name)}, {html.escape(Path(args.token_scores).name)}</small>
{legend}
<pre style="padding:12px;border:1px solid #eee;border-radius:8px;background:#fafafa">{''.join(parts)}</pre>

<h3 style="margin-top:20px">Items (original order)</h3>
{''.join(tbl)}
"""
    Path(args.out_html).write_text(html_doc, encoding="utf-8")

    # Save TSVs
    Path(args.out_sent_tsv).parent.mkdir(parents=True, exist_ok=True)
    df_sent.to_csv(args.out_sent_tsv, sep="\t", index=False)
    df_sec.to_csv(args.out_sec_tsv, sep="\t", index=False)
    df_lab.to_csv(args.out_label_tsv, sep="\t", index=False)
    df_unmatched.to_csv(args.out_unmatched_tsv, sep="\t", index=False)

    print(f"HTML → {args.out_html}")
    print(f"Sentences TSV → {args.out_sent_tsv}")
    print(f"Sections TSV → {args.out_sec_tsv}")
    print(f"Labels TSV → {args.out_label_tsv}")
    print(f"Unmatched TSV → {args.out_unmatched_tsv}")


if __name__ == "__main__":
    main()
