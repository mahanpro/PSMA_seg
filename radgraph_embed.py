"""
Usage:
python radgraph_embed.py --in_clean_jsonl cache/text/clean_reports.jsonl \
                         --out_dir cache/text \
                         --hf_id microsoft/BiomedVLP-CXR-BERT-specialized
"""

# coding: utf-8

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Optional


def embed_one(
    report_text: str, hf_id: str, device: Optional[str], max_length: int
) -> Dict[str, Any]:
    import torch
    from transformers import BertTokenizerFast, AutoModel
    from radgraph import RadGraph

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")

    rg = RadGraph(model_type="modern-radgraph-xl")
    ann = rg([report_text])["0"]
    print("ann: ", ann)
    # Align to word-level (space split) tokens
    space_tokens = ann["text"].split()

    tok = BertTokenizerFast.from_pretrained(hf_id, trust_remote_code=True)
    mdl = AutoModel.from_pretrained(hf_id, trust_remote_code=True).to(dev)
    mdl.eval()

    enc = tok(
        space_tokens,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    # enc = {k: v.to(dev) for k, v in enc.items()}
    enc = enc.to(dev)

    with torch.no_grad():
        hs = mdl(**enc).last_hidden_state[0]  # [T_wp, H]

    print("type(hs): ", type(hs))
    print("hs.shape: ", hs.shape)
    word_ids = enc.word_ids()
    # ----------------------------------------------------------------------- log subwords
    pieces = tok.convert_ids_to_tokens(enc["input_ids"][0].tolist())
    for i, (p, wid) in enumerate(zip(pieces, word_ids)):
        src = None if wid is None else space_tokens[wid]
        print(f"{i:3d}  {p:15s}  word_id={str(wid):>3s}  word={src}")
    print("len(pieces): ", len(pieces))
    print("len(word_ids): ", len(word_ids))
    print("hs.shape[0]: ", hs.shape[0])
    # -----------------------------------------------------------------------
    print("type(word_ids): ", type(word_ids))
    print("word_ids: ", word_ids)
    valid_wids = [w for w in word_ids if w is not None]
    W = (max(valid_wids) + 1) if valid_wids else 0

    # Mean-pool wordpieces → words
    H = hs.shape[-1]  # H = dimension of each vector embedding
    word_vecs = []
    for i in range(W):
        idxs = [j for j, w in enumerate(word_ids) if w == i]
        if idxs:
            word_vecs.append(hs[idxs].mean(0))
        else:
            word_vecs.append(hs.new_zeros(H))

    # print("word_vecs[0].shape: ", word_vecs[0].shape) # 768 = hs.shape[-1]
    print("len(word_vecs): ", len(word_vecs))
    # Aggregate entity spans over word indices
    entity_vecs: Dict[str, Any] = {}
    entity_meta: Dict[str, Any] = {}
    for eid, entity in ann["entities"].items():
        i0 = int(entity.get("start_ix", 0))
        i1 = int(entity.get("end_ix", i0))
        if W == 0 or i1 < i0:
            vec = hs.new_zeros(H)
        else:
            i0 = max(0, i0)
            i1 = min(W - 1, i1)
            vec = (
                (0.5 * (word_vecs[i0] + word_vecs[i1]))
                if i1 == i0
                else (sum(word_vecs[i0 : i1 + 1]) / (i1 - i0 + 1))
            )
        # print("vec.shape: ", vec.shape) # 768 = hs.shape[-1]
        entity_vecs[eid] = vec
        entity_meta[eid] = {
            "tokens": entity.get("tokens"),
            "label": entity.get("label"),
            "start_ix": int(entity.get("start_ix", 0)),
            "end_ix": int(entity.get("end_ix", 0)),
            "relations": entity.get("relations", []),
        }

    # Stack to [N,H] in numeric-eid order if possible
    def _num_key(k: str):
        return (0, int(k)) if k.isdigit() else (1, k)

    eids_sorted = sorted(entity_vecs.keys(), key=_num_key)
    E = (
        torch.stack([entity_vecs[k] for k in eids_sorted])
        if eids_sorted
        else torch.empty(0, H)
    )

    return {
        "hidden_dim": int(H),
        "entity_meta": entity_meta,
        "eids_sorted": eids_sorted,
        "embeddings": E,  # [N,H] tensor
        "rg_model": "modern-radgraph-xl",
        "hf_id": hf_id,
        "num_entities": int(E.shape[0]),
    }


def main():
    ap = argparse.ArgumentParser(description="RadGraph-XL + HF entity embeddings")
    ap.add_argument(
        "--in_clean_jsonl",
        type=str,
        required=True,
        help="clean_reports.jsonl ({id, clean, ...})",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default="cache/text",
        help="Output directory for JSON/PT artifacts + index.jsonl",
    )
    ap.add_argument(
        "--hf_id",
        type=str,
        default="microsoft/BiomedVLP-CXR-BERT-specialized",
        help="HF encoder id used with BertTokenizerFast",
    )
    ap.add_argument(
        "--device", type=str, default=None, help="cuda / cpu (default: auto)"
    )
    ap.add_argument("--max_length", type=int, default=512)
    args = ap.parse_args()

    inp = Path(args.in_clean_jsonl)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "index.jsonl"

    n = 0
    import torch

    with open(inp, "r", encoding="utf-8") as fin, open(
        index_path, "w", encoding="utf-8"
    ) as findex:
        for line in fin:
            if not line.strip():
                continue
            row = json.loads(line)
            rid = row["ID"]
            text = row.get("clean") or row.get("raw") or ""
            if not text.strip():
                # still write an empty artifact set
                meta_path = out_dir / f"{rid}.json"
                torch_path = out_dir / f"{rid}_entities.pt"
                json.dump(
                    {
                        "ID": rid,
                        "entity_meta": {},
                        "hf_id": args.hf_id,
                        "rg_model": "modern-radgraph-xl",
                        "num_entities": 0,
                        "hidden_dim": 0,
                    },
                    open(meta_path, "w", encoding="utf-8"),
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

            pack = embed_one(
                text, hf_id=args.hf_id, device=args.device, max_length=args.max_length
            )

            meta = {
                "id": rid,
                "hf_id": pack["hf_id"],
                "rg_model": pack["rg_model"],
                "num_entities": pack["num_entities"],
                "hidden_dim": pack["hidden_dim"],
                "entity_meta": pack["entity_meta"],  # lightweight; no raw text
            }

            meta_path = out_dir / f"{rid}.json"
            torch_path = out_dir / f"{rid}_entities.pt"

            with open(meta_path, "w", encoding="utf-8") as fjson:
                json.dump(meta, fjson, ensure_ascii=False)

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
