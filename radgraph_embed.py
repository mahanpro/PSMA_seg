"""
Usage:
python radgraph_embed.py --in_clean_jsonl cache/text/clean_reports.jsonl \
                         --out_dir cache/text \
                         --hf_id microsoft/BiomedVLP-CXR-BERT-specialized
"""

# coding: utf-8
import argparse, json, math, re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---- Negation handling ----
NEGATORS_ONE = {"no", "without", "absent", "negative"}
NEGATORS_TWO = {("absence", "of")}

# ---- SUVmax extraction ----
SUVMAX_CLAUSE = re.compile(
    r"(?i)\bSUV\s*max\b(?:\s*[:=])?\s*(?:(?:up\s*to|upto)\s*)?"
    r"(?:\d+(?:\s*\.\s*\d+)?|[<>]\s*\d+(?:\s*\.\s*\d+)?)"
    r"(?:\s*(?:and|,)\s*(?:[<>]?\s*\d+(?:\s*\.\s*\d+)?))*"
)
NUM_F = re.compile(r"[<>]?\s*\d+(?:\s*\.\s*\d+)?")


def extract_suv_values(text: str) -> List[float]:
    vals: List[float] = []
    for m in SUVMAX_CLAUSE.finditer(text):
        clause = m.group(0)
        # local numeric repairs
        clause = re.sub(r"(\d)\s*\.\s*(\d)", r"\1.\2", clause)  # "7 . 8" -> "7.8"
        clause = re.sub(
            r"=\s*\d+\s+(\d+\.\d+)", r"= \1", clause
        )  # "= 7 78.3" -> "= 78.3"
        for n in NUM_F.findall(clause):
            n2 = n.replace(" ", "").replace("<", "").replace(">", "")
            try:
                vals.append(float(n2))
            except Exception:
                pass
    return vals


# ---- Core embedding helpers ----
def _mean_pool_wordpieces_to_words(enc, hs):
    """Return list of [H] tensors, one per original whitespace-split token."""
    import torch

    word_ids = enc.word_ids()
    valid = [w for w in word_ids if w is not None]
    W = (max(valid) + 1) if valid else 0
    H = hs.shape[-1]
    word_vecs: List[torch.Tensor] = []
    for i in range(W):
        idxs = [j for j, w in enumerate(word_ids) if w == i]
        if idxs:
            word_vecs.append(hs[idxs].mean(0))
        else:
            word_vecs.append(hs.new_zeros(H))
    return word_vecs  # len=W, each [H]


def _widen_for_negation(space_tokens: List[str], i0: int, i1: int) -> Tuple[int, int]:
    """Include immediate negators to the left of the entity span."""
    if i0 <= 0:
        return i0, i1
    t0 = space_tokens[i0 - 1].lower()
    if t0 in NEGATORS_ONE:
        i0 = i0 - 1
        # also include bigram 'absence of' if present even further left
        if i0 >= 2:
            if (
                space_tokens[i0 - 2].lower(),
                space_tokens[i0 - 1].lower(),
            ) in NEGATORS_TWO:
                i0 = i0 - 2
    elif i0 >= 2 and (space_tokens[i0 - 2].lower(), t0) in NEGATORS_TWO:
        i0 = i0 - 2
    return i0, i1


def embed_one(report_text: str, hf_id: str, device: Optional[str]) -> Dict[str, Any]:
    import torch
    from transformers import BertTokenizerFast, AutoModel
    from radgraph import RadGraph

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # 1) RadGraph annotations
    rg = RadGraph(model="modern-radgraph-xl")
    ann = rg([report_text])["0"]  # string key "0"

    space_tokens = ann["text"].split()

    # 2) HF encoder for embeddings (mean-pooled to whitespace tokens)
    tok = BertTokenizerFast.from_pretrained(hf_id, trust_remote_code=True)
    mdl = AutoModel.from_pretrained(hf_id, trust_remote_code=True).to(dev).eval()

    enc = tok(
        space_tokens,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(dev)
    with torch.no_grad():
        hs = mdl(**enc).last_hidden_state[0]  # [Twp, H]
    word_vecs = _mean_pool_wordpieces_to_words(enc, hs)
    H = hs.shape[-1]
    W = len(word_vecs)

    # 3) Build entity vectors with negation-aware span widening
    entity_vecs: Dict[str, Any] = {}
    entity_meta: Dict[str, Any] = {}

    for eid, entity in ann.get("entities", {}).items():
        i0 = int(entity.get("start_ix", 0))
        i1 = int(entity.get("end_ix", i0))
        label = str(entity.get("label", "")).lower()

        if (
            ("definitely absent" in label)
            or ("uncertain" in label)
            or ("negative" in label)
        ):
            i0, i1 = _widen_for_negation(space_tokens, i0, i1)

        # clamp and pool
        if W == 0:
            vec = hs.new_zeros(H)
        else:
            i0c = max(0, min(W - 1, i0))
            i1c = max(0, min(W - 1, i1))
            lo, hi = (i0c, i1c) if i0c <= i1c else (i1c, i0c)
            vec = sum(word_vecs[lo : hi + 1]) / max(1, (hi - lo + 1))

        # reconstruct tokens over widened span for readability
        toks = (
            " ".join(space_tokens[max(0, i0) : min(W - 1, i1) + 1])
            if W > 0
            else str(entity.get("tokens"))
        )

        entity_vecs[eid] = vec
        entity_meta[eid] = {
            "tokens": toks,
            "label": entity.get("label"),
            "start_ix": int(entity.get("start_ix", 0)),
            "end_ix": int(entity.get("end_ix", 0)),
            "relations": entity.get("relations", []),
        }

    # 4) Stack embeddings and sort eids by original start_ix for determinism
    def start_ix_of(k: str) -> int:
        try:
            return int(entity_meta[k].get("start_ix", 0))
        except Exception:
            return 0

    eids_sorted = sorted(entity_vecs.keys(), key=start_ix_of)

    E = (
        torch.stack([entity_vecs[k] for k in eids_sorted])
        if eids_sorted
        else torch.empty(0, H, device=hs.device)
    )

    # 5) Global SUVmax values (no per-section association)
    suv_all = extract_suv_values(report_text)

    return {
        "hidden_dim": int(H),
        "num_entities": int(E.shape[0]),
        "entity_meta": entity_meta,
        "eids_sorted": eids_sorted,
        "embeddings": E,  # [N, H] torch tensor
        "suvmax_all": suv_all,  # list[float]
        "rg_model": "modern-radgraph-xl",
        "hf_id": hf_id,
    }


def main():
    ap = argparse.ArgumentParser(description="RadGraph + HF embeddings (pruned)")
    ap.add_argument(
        "--in_clean_jsonl",
        type=str,
        required=True,
        help="clean_reports.jsonl ({ID, clean})",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default="cache/text",
        help="Output dir for {ID}.json, {ID}_entities.pt, and index.jsonl",
    )
    ap.add_argument(
        "--hf_id", type=str, default="microsoft/BiomedVLP-CXR-BERT-specialized"
    )
    ap.add_argument(
        "--device", type=str, default=None, help="cuda / cpu (default: auto)"
    )
    args = ap.parse_args()

    inp = Path(args.in_clean_jsonl)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "index.jsonl"

    import torch

    n = 0
    with open(inp, "r", encoding="utf-8") as fin, open(
        index_path, "w", encoding="utf-8"
    ) as findex:
        for line in fin:
            if not line.strip():
                continue
            row = json.loads(line)
            rid = row.get("ID") or row.get("id") or row.get("Id")
            text = (
                row.get("clean") or row.get("Description") or row.get("raw") or ""
            ).strip()

            meta_path = out_dir / f"{rid}.json"
            torch_path = out_dir / f"{rid}_entities.pt"

            if not rid or not text:
                json.dump(
                    {"id": rid, "num_entities": 0, "hidden_dim": 0, "suvmax_all": []},
                    open(meta_path, "w", encoding="utf-8"),
                    ensure_ascii=False,
                )
                torch.save({"eids": [], "embeddings": torch.empty(0, 0)}, torch_path)
                findex.write(
                    json.dumps(
                        {
                            "id": rid,
                            "clean_json": str(meta_path),
                            "entities_pt": str(torch_path),
                            "hidden_dim": 0,
                            "num_entities": 0,
                        }
                    )
                    + "\n"
                )
                continue

            pack = embed_one(text, hf_id=args.hf_id, device=args.device)

            # write JSON meta
            meta = {
                "id": rid,
                "hf_id": pack["hf_id"],
                "rg_model": pack["rg_model"],
                "num_entities": pack["num_entities"],
                "hidden_dim": pack["hidden_dim"],
                "entity_meta": pack["entity_meta"],
                "suvmax_all": pack["suvmax_all"],
            }
            with open(meta_path, "w", encoding="utf-8") as fjson:
                json.dump(meta, fjson, ensure_ascii=False)

            # write embeddings tensor
            torch.save(
                {"eids": pack["eids_sorted"], "embeddings": pack["embeddings"]},
                torch_path,
            )

            findex.write(
                json.dumps(
                    {
                        "id": rid,
                        "clean_json": str(meta_path),
                        "entities_pt": str(torch_path),
                        "hidden_dim": pack["hidden_dim"],
                        "num_entities": pack["num_entities"],
                    }
                )
                + "\n"
            )

            n += 1

    print(f"Processed {n} reports → {out_dir} (index: {index_path})")


if __name__ == "__main__":
    main()
