import json, torch
from pathlib import Path
from typing import Dict, Tuple, Optional
import torch.nn.functional as F

"""
Utilities to load per-patient text embeddings for the four ablations:

- "image": returns None (no text)
- "radgraph": expects index.jsonl produced by your radgraph_embed.py
- "gpt":      expects index_gpt.jsonl produced by your gpt_embed.py
- "msraw":    expects msraw_index.jsonl produced by tools/build_msraw_index.py
"""


def _sanitize_tokens(E: torch.Tensor) -> torch.Tensor:
    # Remove NaN/Inf and bound extremes, then L2-normalize per token.
    E = torch.nan_to_num(E, nan=0.0, posinf=1e4, neginf=-1e4)
    n = E.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return E / n


def _load_index_lines(p: Path):
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


_LUT_CACHE = {}


def build_lookup(index_path: Path) -> Dict[str, Dict[str, str]]:
    p = str(index_path.resolve())
    if p in _LUT_CACHE:
        return _LUT_CACHE[p]
    lut = {}
    for row in _load_index_lines(index_path):
        rid = str(row.get("id"))
        lut[rid] = {
            "clean_json": row.get("clean_json"),
            "entities_pt": row.get("entities_pt"),
            "hidden_dim": row.get("hidden_dim", 0),
            "num_entities": row.get("num_entities", 0),
        }
    _LUT_CACHE[p] = lut
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
        rec = idx.get(f"{int(rid):04d}")
        if not rec:
            return None, None
        pt = torch.load(rec["entities_pt"], map_location="cpu")
        E = pt["embeddings"]
        E = _sanitize_tokens(E)
        return E.float(), None

    if modality == "gpt":
        idx = build_lookup(roots["gpt"] / "index_gpt.jsonl")
        rec = idx.get(f"{int(rid):04d}")
        if not rec:
            return None, None
        pt = torch.load(rec["entities_pt"], map_location="cpu")
        E = pt["embeddings"]
        E = _sanitize_tokens(E)
        return E.float(), None

    if modality == "gpt_raw":
        # index_gpt_raw.jsonl lives under the gpt_raw root
        idx = build_lookup(roots["gpt_raw"] / "index_gpt_raw.jsonl")
        rec = idx.get(str(rid)) or idx.get(f"{int(rid):04d}")
        if not rec:
            return None, None
        pt = torch.load(rec["entities_pt"], map_location="cpu")
        E = pt.get("embeddings")  # (L, H)
        E = _sanitize_tokens(E)
        if E is None:
            return None, None
        return E.float(), None

    if modality == "msraw":
        idx_path = _first_existing(
            roots["msraw"] / "msraw_index.jsonl",
        )
        if not idx_path:
            return None, None
        idx = build_lookup(idx_path)
        rec = idx.get(str(rid)) or idx.get(f"{int(rid):04d}")
        if not rec:
            return None, None
        pack = torch.load(rec["entities_pt"], map_location="cpu")
        # builder saves token-level embeddings + mask
        tokens = pack.get("tokens")  # (L, Ct)
        mask = pack.get("mask")  # (L,)
        if tokens is None:
            return None, None
        tokens = _sanitize_tokens(tokens)
        tokens = tokens.float()
        if mask is not None:
            mask = mask.bool()
        return tokens, mask

    raise ValueError(f"Unknown text modality: {modality}")
