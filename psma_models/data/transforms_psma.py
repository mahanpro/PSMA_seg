from monai.transforms.compose import Compose
from monai.transforms.io.dictionary import LoadImaged
from monai.transforms.utility.dictionary import EnsureChannelFirstd, EnsureTyped
from monai.transforms.intensity.dictionary import ScaleIntensityRanged
from monai.transforms.spatial.dictionary import (
    Orientationd,
    RandAffined,
)
from monai.transforms.croppad.dictionary import CropForegroundd
from monai.transforms.utility.dictionary import ConcatItemsd, DeleteItemsd

from monai.transforms.spatial.dictionary import Rand3DElasticd
from monai.transforms.intensity.dictionary import (
    RandAdjustContrastd,
    RandGaussianNoised,
)
from textfusion.transforms import AddTextTokensd


def make_transforms(
    text_root: str, train: bool, include_mask: bool = True, force_orient: bool = False
):
    # images are already on the PET grid; Orientationd is cheap & safe if headers vary
    keys = ["CT", "PT"] + (["GT"] if include_mask else [])
    aug = []
    if train:
        aug = [
            RandAffined(
                keys=keys,
                mode=("bilinear", "bilinear", "nearest")[: len(keys)],
                prob=0.5,
                translate_range=(8, 8, 8),
                rotate_range=(0, 0, 0.26),
                scale_range=(0.1, 0.1, 0.1),
            ),
            Rand3DElasticd(
                keys=keys, sigma_range=(0.0, 1.0), magnitude_range=(0.0, 1.0), prob=0.3
            ),
            RandAdjustContrastd(keys=["CT", "PT"], prob=0.3, gamma=(0.7, 1.5)),
            RandGaussianNoised(keys=["CT", "PT"], prob=0.5),
        ]

    ops = [
        LoadImaged(keys=keys, image_only=False),
        EnsureChannelFirstd(keys=keys),
        CropForegroundd(keys=keys, source_key="CT"),
    ]
    if force_orient:
        ops.append(Orientationd(keys=keys, axcodes="RAS"))
    ops += (
        [
            ScaleIntensityRanged(
                keys=["CT"], a_min=-1024, a_max=1024, b_min=0, b_max=1, clip=True
            ),
            ConcatItemsd(keys=["CT", "PT"], name="CTPT", dim=0),
            DeleteItemsd(keys=["CT", "PT"]),
        ]
        + aug
        + [
            # Insert tokens from cache/text_msraw (uses d['ID'])
            AddTextTokensd(keys="ID", text_root=text_root, allow_missing=False),
            EnsureTyped(
                keys=["CTPT"] + (["GT"] if include_mask else []) + ["TXT", "TXT_MASK"]
            ),
        ]
    )
    return Compose(ops)
