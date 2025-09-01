from pathlib import Path
import torch, json


def tokens_paths_for_id(text_root: Path, rid: str):
    """Return (meta_path, pt_path) for MS-RAW artifacts."""
    meta_path = text_root / f"{rid}_msraw.json"
    torch_path = text_root / f"{rid}_msraw_tokens.pt"
    return meta_path, torch_path


def load_tokens_for_id(text_root, rid: str):
    """
    Load token embeddings + mask for given ID.
    Returns (tokens: (L,Ct) float32 torch.Tensor, mask: (L,) bool torch.Tensor)
    Raises FileNotFoundError if missing.
    """
    _, torch_path = tokens_paths_for_id(Path(text_root), rid)
    pack = torch.load(torch_path, map_location="cpu")
    tokens = pack.get("tokens")
    mask = pack.get("mask")
    if tokens is None or mask is None:
        raise RuntimeError(f"Corrupt tokens file: {torch_path}")
    if tokens.dtype != torch.float32:
        tokens = tokens.float()
    if mask.dtype != torch.bool:
        mask = mask.bool()
    return tokens, mask
