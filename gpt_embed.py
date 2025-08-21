"""
Usage:
export OPENAI_API_KEY=sk-...
pip install openai torch
python gpt_embed.py --in_clean_jsonl cache/text/clean_reports.jsonl \
                    --out_dir cache/text_gpt \
                    --gpt_model gpt-4o-mini \
                    --embed_model text-embedding-3-small
"""

# coding: utf-8
import argparse, json, re
from pathlib import Path
from typing import Any, Dict, List

# ---- SUVmax regex fallback (same as in radgraph script) ----
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
        clause = re.sub(r"(\d)\s*\.\s*(\d)", r"\1.\2", clause)
        clause = re.sub(r"=\s*\d+\s+(\d+\.\d+)", r"= \1", clause)
        for n in NUM_F.findall(clause):
            n2 = n.replace(" ", "").replace("<", "").replace(">", "")
            try:
                vals.append(float(n2))
            except Exception:
                pass
    return vals


GPT_SYSTEM = (
    "You are an information extraction tool for PSMA PET reports. "
    "Return STRICT JSON. Capture negation by merging nearby cues (e.g., "
    "'No abnormal ... lesion' should be one absent Observation). "
    "Do not invent text spans."
)

GPT_USER_TMPL = """Extract entities and SUVmax values from this report.

Text:
<<<
{TEXT}
>>>

Return JSON with:
- "entities": list of objects, each:
  {{"text": string, "label": string, "start_ix": int, "end_ix": int}}
  Labels may be: "Anatomy::definitely present", "Observation::definitely present",
                 "Observation::definitely absent", "Observation::uncertain",
                 "Observation::measurement::definitely present"
  The indices are whitespace-token indices (0-based) over text.split().
  If a phrase is negated (e.g., starts with 'No' / 'without' / 'absence of'), include the negator in 'text' and set label to an 'absent' or appropriate form.
- "suvmax_all": list of numbers extracted from any 'SUVmax' mention in the text.
Only output JSON, no extra commentary.
"""


def call_gpt_extract(client, model: str, text: str) -> Dict[str, Any]:
    msg_user = GPT_USER_TMPL.format(TEXT=text)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": GPT_SYSTEM},
            {"role": "user", "content": msg_user},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    content = resp.choices[0].message.content
    try:
        data = json.loads(content)
    except Exception:
        data = {"entities": [], "suvmax_all": []}
    # Fallback for SUVs if missing
    if not data.get("suvmax_all"):
        data["suvmax_all"] = extract_suv_values(text)
    return data


def main():
    ap = argparse.ArgumentParser(
        description="OpenAI GPT entity+embedding pipeline (ablation)"
    )
    ap.add_argument(
        "--in_clean_jsonl",
        type=str,
        required=True,
        help="clean_reports.jsonl ({ID, clean})",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default="cache/text_gpt",
        help="Output dir for {ID}.json, {ID}_entities.pt, and index.jsonl",
    )
    ap.add_argument("--gpt_model", type=str, default="gpt-5-2025-08-07")
    ap.add_argument("--embed_model", type=str, default="text-embedding-3-small")
    args = ap.parse_args()

    from openai import OpenAI
    import torch

    client = OpenAI()

    inp = Path(args.in_clean_jsonl)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "index.jsonl"

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
                    {
                        "id": rid,
                        "gpt_model": args.gpt_model,
                        "embedding_model": args.embed_model,
                        "num_entities": 0,
                        "hidden_dim": 0,
                        "suvmax_all": [],
                    },
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

            # 1) Extract entities + SUVs with GPT
            data = call_gpt_extract(client, args.gpt_model, text)
            entities = data.get("entities", [])
            suv_all = data.get("suvmax_all", [])

            # 2) Prepare strings for embedding (entity texts)
            entity_texts = [str(e.get("text", "")) for e in entities]
            if entity_texts:
                emb_resp = client.embeddings.create(
                    model=args.embed_model, input=entity_texts
                )
                vectors = [item.embedding for item in emb_resp.data]
                import numpy as np
                import torch

                E = torch.tensor(np.array(vectors), dtype=torch.float32)
                hidden_dim = int(E.shape[1])
                eids_sorted = [str(i + 1) for i in range(len(entity_texts))]
                torch.save({"eids": eids_sorted, "embeddings": E}, torch_path)
                num_entities = len(entity_texts)
            else:
                import torch

                E = torch.empty(0, 0)
                torch.save({"eids": [], "embeddings": E}, torch_path)
                hidden_dim = 0
                num_entities = 0

            # 3) Save JSON meta (keep raw positions if model provided them)
            entity_meta = {}
            for i, ent in enumerate(entities, start=1):
                entity_meta[str(i)] = {
                    "tokens": ent.get("text", ""),
                    "label": ent.get("label", ""),
                    "start_ix": int(ent.get("start_ix", 0)),
                    "end_ix": int(ent.get("end_ix", 0)),
                    "relations": [],
                }

            meta = {
                "id": rid,
                "gpt_model": args.gpt_model,
                "embedding_model": args.embed_model,
                "num_entities": num_entities,
                "hidden_dim": hidden_dim,
                "entity_meta": entity_meta,
                "suvmax_all": suv_all,
            }
            with open(meta_path, "w", encoding="utf-8") as fjson:
                json.dump(meta, fjson, ensure_ascii=False)

            findex.write(
                json.dumps(
                    {
                        "id": rid,
                        "clean_json": str(meta_path),
                        "entities_pt": str(torch_path),
                        "hidden_dim": hidden_dim,
                        "num_entities": num_entities,
                    }
                )
                + "\n"
            )

            n += 1

    print(f"Processed {n} reports → {out_dir} (index: {index_path})")


if __name__ == "__main__":
    main()
