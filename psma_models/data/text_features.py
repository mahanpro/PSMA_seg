import json, torch
from pathlib import Path
from typing import Dict, Tuple, Optional

"""
Utilities to load per-patient text embeddings for the four ablations:

- "image": returns None (no text)
- "radgraph": expects index.jsonl produced by your radgraph_embed.py
- "gpt":      expects index_gpt.jsonl produced by your gpt_embed.py
- "msraw":    expects msraw_index.jsonl produced by tools/build_msraw_index.py
"""


def _load_index_lines(p: Path):
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def build_lookup(index_path: Path) -> Dict[str, Dict[str, str]]:
    lut = {}
    for row in _load_index_lines(index_path):
        rid = str(row.get("id"))
        lut[rid] = {
            "clean_json": row.get("clean_json"),
            "entities_pt": row.get("entities_pt"),
            "hidden_dim": row.get("hidden_dim", 0),
            "num_entities": row.get("num_entities", 0),
        }
    return lut


def _first_existing(*candidates: Path) -> Optional[Path]:
    for p in candidates:
        if p and p.exists():
            return p
    return None


def load_tokens_for_id(
    modality: str, rid: str, roots: Dict[str, Path]
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if modality == "image":
        return None, None

    if modality == "radgraph":
        idx = build_lookup(roots["radgraph"] / "index.jsonl")
        rec = idx.get(rid)
        if not rec:
            return None, None
        pt = torch.load(rec["entities_pt"], map_location="cpu")
        E = pt["embeddings"]
        return E.float(), None

    if modality == "gpt":
        idx = build_lookup(roots["gpt"] / "index_gpt.jsonl")
        rec = idx.get(rid)
        if not rec:
            return None, None
        pt = torch.load(rec["entities_pt"], map_location="cpu")
        E = pt["embeddings"]
        return E.float(), None

    if modality == "msraw":
        # accept either new or old filename
        idx_path = _first_existing(
            roots["msraw"] / "msraw_index.jsonl",
            roots["msraw"] / "index_msraw.jsonl",
        )
        if not idx_path:
            return None, None
        idx = build_lookup(idx_path)
        rec = idx.get(rid)
        if not rec:
            return None, None
        pt = torch.load(rec["entities_pt"], map_location="cpu")
        E = pt["embeddings"]  # (L, H)
        return E.float(), None

    raise ValueError(f"Unknown text modality: {modality}")
