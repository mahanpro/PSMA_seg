from typing import List, Dict, Any
import torch
from monai.data.utils import no_collation


from typing import List, Dict, Any
import torch
from monai.data.utils import no_collation


def text_list_data_collate(batch: List[Dict[str, Any]]):
    """
    Like MONAI's list_data_collate, but:
      - pads TXT (L,Ct) → (B,Lmax,Ct)
      - builds TXT_MASK as PyTorch key_padding_mask (B,Lmax) with True==PAD
    """
    if not isinstance(batch, list):
        return no_collation(batch)

    # Pull out text fields
    txt_list, rest = [], []
    for d in batch:
        txt_list.append(d.pop("TXT", None))
        rest.append(d)

    from monai.data.utils import list_data_collate as monai_collate

    out = monai_collate(rest)

    if any(t is not None for t in txt_list):
        # lengths and dim check
        lengths = [t.shape[0] if t is not None else 0 for t in txt_list]
        Ct = max((t.shape[1] for t in txt_list if t is not None), default=0)
        Lmax = max(lengths) if lengths else 0
        B = len(txt_list)
        txt_pad = torch.zeros((B, Lmax, Ct), dtype=torch.float32)
        pad_mask = torch.zeros((B, Lmax), dtype=torch.bool)  # True==PAD

        for i, t in enumerate(txt_list):
            if t is None or t.numel() == 0:
                pad_mask[i, :] = True
                continue
            L = t.shape[0]
            txt_pad[i, :L, :] = t
            if L < Lmax:
                pad_mask[i, L:] = True

        out["TXT"] = txt_pad
        out["TXT_MASK"] = pad_mask

    else:
        out["TXT"] = None
        out["TXT_MASK"] = None

    return out
