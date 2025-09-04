from typing import List, Dict, Any
import torch
from monai.data.utils import no_collation


from typing import List, Dict, Any
import torch
from monai.data.utils import no_collation


def text_list_data_collate(batch: List[Dict[str, Any]]):
    from monai.data.utils import list_data_collate as monai_collate

    if not isinstance(batch, list):
        return no_collation(batch)
    # Flatten nested lists (from RandCropByPosNegLabeld num_samples>1)
    flat = []
    stack = [batch]
    while stack:
        cur = stack.pop()
        for x in cur:
            if isinstance(x, list):
                stack.append(x)
            else:
                flat.append(x)
    batch = flat

    txt_list, rest = [], []
    for d in batch:
        # remove text fields so MONAI doesn't try to collate Nones
        txt_list.append(d.pop("TXT", None))
        d.pop("TXT_MASK", None)
        rest.append(d)

    out = monai_collate(rest)

    if any(t is not None for t in txt_list):
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
        # image-only path: still return placeholders so your training loop works
        out["TXT"] = None
        out["TXT_MASK"] = None

    return out
