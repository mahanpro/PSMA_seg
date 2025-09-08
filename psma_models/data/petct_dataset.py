import torch
from typing import Dict, Any
from monai.transforms.compose import Compose
from monai.transforms.io.dictionary import LoadImaged
from monai.transforms.utility.dictionary import EnsureChannelFirstd, EnsureTyped
from monai.transforms.intensity.dictionary import ScaleIntensityRanged
from monai.data.dataset import Dataset
from .text_features import load_tokens_for_id

from monai.transforms.spatial.dictionary import (
    Orientationd,
    RandAffined,
)
from monai.transforms.croppad.dictionary import CropForegroundd
from monai.transforms.utility.dictionary import ConcatItemsd, DeleteItemsd

from monai.transforms.spatial.dictionary import Rand3DElasticd
from monai.transforms.croppad.dictionary import (
    RandCropByPosNegLabeld,
    RandSpatialCropd,
    ResizeWithPadOrCropd,
)
from monai.transforms.intensity.dictionary import (
    RandAdjustContrastd,
    RandGaussianNoised,
)

# --------- NEW: no text injection here; dataset class handles text ----------


def make_transforms(
    train: bool,
    include_mask: bool = True,
    force_orient: bool = False,
    pet_clip: float = 35.0,
    patch: int | None = None,  # <- optional training patch (see below)
):
    keys = ["CT", "PT"] + (["GT"] if include_mask else [])

    aug = []
    if train:
        aug = [
            # spatial augs on CT, PT, (and GT with nearest)
            RandAffined(
                keys=keys,
                mode=("bilinear", "bilinear", "nearest")[: len(keys)],
                prob=0.5,
                translate_range=(8, 8, 8),
                rotate_range=(0.0, 0.0, 0.26),
                scale_range=(0.10, 0.10, 0.10),
            ),
            Rand3DElasticd(
                keys=keys,
                mode=("bilinear", "bilinear", "nearest")[: len(keys)],
                sigma_range=(0.0, 1.0),
                magnitude_range=(0.0, 1.0),
                prob=0.3,
            ),
            # light intensity augs per modality
            RandAdjustContrastd(keys=["CT", "PT"], prob=0.3, gamma=(0.7, 1.5)),
            RandGaussianNoised(keys=["CT", "PT"], prob=0.5),
        ]

    ops = [
        LoadImaged(keys=keys, image_only=False),
        EnsureChannelFirstd(keys=keys),
        # drop -1000 HU padding (uses CT as body mask, deterministic)
        CropForegroundd(
            keys=keys, source_key="CT", select_fn=lambda x: x > -900, margin=8
        ),
    ]
    if force_orient:
        ops.append(Orientationd(keys=keys, axcodes="RAS"))

    # per-modality scaling BEFORE aug keeps HU/SUV semantics stable
    ops += [
        ScaleIntensityRanged(
            keys=["CT"], a_min=-1024, a_max=1024, b_min=0.0, b_max=1.0, clip=True
        ),
        ScaleIntensityRanged(
            keys=["PT"],
            a_min=0.0,
            a_max=float(pet_clip),
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
    ]

    # train-time patch crop (prefer positive-biased sampling if GT present)
    if train and patch:
        if include_mask:
            ops += [
                RandCropByPosNegLabeld(
                    keys=keys,
                    label_key="GT",
                    spatial_size=(patch, patch, patch),
                    pos=3,
                    neg=1,
                    num_samples=2,
                    image_key="CT",  # either CT or PT is fine for spacing
                    image_threshold=-900,  # ignore air
                    allow_smaller=True,
                ),
                ResizeWithPadOrCropd(
                    keys=keys,
                    spatial_size=(patch, patch, patch),
                    mode="constant",
                ),
            ]
        else:
            ops += [  # fallback if GT missing for some ablations
                RandSpatialCropd(
                    keys=keys,
                    roi_size=(patch, patch, patch),
                    random_center=True,
                    random_size=False,
                ),
                ResizeWithPadOrCropd(
                    keys=keys, spatial_size=(patch, patch, patch), mode="constant"
                ),
            ]

    # now do the augs while CT/PT still exist
    ops += aug

    # finally build the 2-channel tensor the model expects
    ops += [
        ConcatItemsd(keys=["CT", "PT"], name="CTPT", dim=0),
        DeleteItemsd(keys=["CT", "PT"]),
        EnsureTyped(keys=["CTPT"] + (["GT"] if include_mask else [])),
    ]
    return Compose(ops)


# -------------------- Dataset with conditional text loading --------------------

from .text_features import load_tokens_for_id


class PSMAJSONDataset(Dataset):
    """
    Expects list[dict]: keys = ID, CT, PT, GT (optional).
    text_modality: "image" | "radgraph" | "gpt" | "msraw"
    text_roots: dict like {"radgraph": Path(...), "gpt": Path(...), "msraw": Path(...)}
    """

    def __init__(
        self, data, transforms, text_modality="image", text_roots=None, max_tokens=1024
    ):
        super().__init__(data=data, transform=transforms)
        self.text_modality = text_modality
        self.text_roots = text_roots or {}
        self.max_tokens = max_tokens

    def _clip_tokens(self, tok: torch.Tensor | None):
        if tok is None:
            return None
        L = tok.shape[0]
        if L <= self.max_tokens:
            return tok
        head = 256
        tail = self.max_tokens - head
        return torch.cat([tok[:head], tok[-tail:]], dim=0)

    def __getitem__(self, idx):
        out = super().__getitem__(idx)  # dict OR list[dict]
        rid = str(self.data[idx]["ID"])

        def _attach_meta(d):
            d["ID"] = rid
            if self.text_modality != "image":
                tokens, mask = load_tokens_for_id(
                    self.text_modality, rid, self.text_roots
                )
                if tokens is not None:
                    tokens = self._clip_tokens(tokens)
                    mask = None
                d["TXT"] = tokens
                d["TXT_MASK"] = mask
            return d

        if isinstance(out, list):
            return [_attach_meta(d) for d in out]
        else:
            return _attach_meta(out)
