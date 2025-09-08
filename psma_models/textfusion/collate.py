from typing import List, Dict, Any, cast
import torch
from monai.data.utils import no_collation


def text_list_data_collate(batch: List[Dict[str, Any]]):
    from monai.data.utils import list_data_collate as monai_collate

    if not isinstance(batch, list):
        return no_collation(batch)

    # Flatten nested lists (from RandCropByPosNegLabeld num_samples>1)
    flat: List[Dict[str, Any]] = []
    stack: List[Any] = [batch]
    while stack:
        cur = stack.pop()
        for x in cur:
            if isinstance(x, list):
                stack.append(x)
            else:
                flat.append(x)
    batch = flat

    txt_list, mask_list, rest = [], [], []
    for d in batch:
        # pull text fields out so collate doesn't choke on Nones / var-len
        txt_list.append(d.pop("TXT", None))  # Tensor[L, Ct] or None
        mask_list.append(d.pop("TXT_MASK", None))  # Bool[L] (True==VALID) or None
        rest.append(d)

    out: Dict[str, Any] = cast(
        Dict[str, Any], monai_collate(rest)
    )  # help static type checker

    if any(t is not None for t in txt_list):
        lengths = [int(t.shape[0]) if t is not None else 0 for t in txt_list]
        Ct = max((int(t.shape[1]) for t in txt_list if t is not None), default=0)
        Lmax = max(lengths) if lengths else 0
        B = len(txt_list)

        txt_pad = torch.zeros((B, Lmax, Ct), dtype=torch.float32)
        # PyTorch MultiheadAttention: key_padding_mask=True means PAD (ignored)
        # https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
        pad_mask = torch.ones((B, Lmax), dtype=torch.bool)  # init as all PAD=True
        # we'll flip valid regions to False below

        for i, (t, m) in enumerate(zip(txt_list, mask_list)):
            if t is None or t.numel() == 0:
                continue
            L = int(t.shape[0])
            txt_pad[i, :L, :] = t

            if m is not None:
                # Your MS-RAW mask marks VALID tokens as True; invert for key_padding_mask.
                vm = m.bool()
                pad_mask[i, :L] = ~vm
            else:
                # No mask provided; only the padded tail is PAD.
                pad_mask[i, :L] = False
                if L < Lmax:
                    pad_mask[i, L:] = True

        out["TXT"] = txt_pad
        out["TXT_MASK"] = pad_mask
    else:
        out["TXT"] = None
        out["TXT_MASK"] = None

    return out
