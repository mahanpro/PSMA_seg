"""
Runs inference on a manifest split, writes NIfTI predictions and optional attention heatmaps
projected to full resolution (upsampling from final decoder stage).
Also computes whole-body 3D Dice.

Usage:
python infer_and_metrics.py \
  --manifest data/dataset.csv --fold 0 \
  --ckpt runs/exp1_text/ models/fold0_ep0300.pth \
  --outdir runs/exp1_text/preds_fold0 \
  --text-modality image|radgraph|gpt|msraw \
  --radgraph-root cache/text --gpt-root cache/text --msraw-root cache/text_msraw
"""

import os, argparse, json
from pathlib import Path
import torch
import numpy as np
import nibabel as nib
from monai.metrics.meandice import DiceMetric
from monai.transforms.compose import Compose
from monai.inferers.utils import sliding_window_inference
from monai.transforms.post.array import AsDiscrete
from monai.data.utils import decollate_batch
from data.petct_dataset import PSMAJSONDataset, make_transforms
from psma_models.data.text_features import load_tokens_for_id
from models.unet3d_text import UNet3DText
from typing import Dict, Any, cast
from nibabel.loadsave import save as nib_save
from nibabel.nifti1 import Nifti1Image
import pandas as pd
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from monai.networks.nets.segresnet import SegResNet
from models.segresnet_text import SegResNetText
from tools.ckpt_utils import load_init_ckpt

from torch.amp.autocast_mode import autocast


class _Wrap(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x, *_):
        return self.net(x), None


def save_nifti_like(array, out_path):
    img = Nifti1Image(array.astype(np.uint8), affine=np.eye(4))
    nib_save(img, str(out_path))


def infer_txt_dim_safely(text_modality, text_roots, id_list, default_if_image=0):
    if text_modality == "image":
        return default_if_image
    for r in id_list[:4]:  # try a few IDs
        tok, _ = load_tokens_for_id(text_modality, str(r["ID"]), text_roots)
        if tok is not None:
            return tok.shape[-1]
    raise RuntimeError(
        f"Could not infer txt_dim for modality={text_modality}; "
        "no token files found for the provided IDs."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument(
        "--text-modality",
        choices=["image", "radgraph", "gpt", "msraw"],
        default="image",
    )
    ap.add_argument("--radgraph-root", default="")
    ap.add_argument("--gpt-root", default="")
    ap.add_argument("--msraw-root", default="")
    ap.add_argument("--infer_patch", type=int, default=192)
    ap.add_argument("--save_heatmaps", action="store_true")
    ap.add_argument("--amp", action="store_true", default=True)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True  # ROI size is fixed, safe to enable

    df = pd.read_csv(args.manifest)
    val_rows = df[df["fold"] == args.fold].to_dict("records")

    t = make_transforms(
        train=False, include_mask=True, force_orient=False, pet_clip=35.0
    )
    text_roots = {
        "radgraph": Path(args.radgraph_root) if args.radgraph_root else Path("."),
        "gpt": Path(args.gpt_root) if args.gpt_root else Path("."),
        "msraw": Path(args.msraw_root) if args.msraw_root else Path("."),
    }
    ds = PSMAJSONDataset(
        val_rows, t, text_modality=args.text_modality, text_roots=text_roots
    )

    txt_dim = infer_txt_dim_safely(
        args.text_modality, text_roots, val_rows, default_if_image=0
    )

    use_text = args.text_modality != "image"

    if use_text:
        model = SegResNetText(
            in_channels=2,
            out_channels=2,
            init_filters=16,
            blocks_down=[1, 2, 2, 4],
            blocks_up=[1, 1, 1],
            txt_dim=txt_dim,
            use_text=True,
            n_heads=4,
        ).to(device)
    else:
        model = SegResNet(
            spatial_dims=3,
            in_channels=2,
            out_channels=2,
            init_filters=16,
            blocks_down=(1, 2, 2, 4),
            blocks_up=(1, 1, 1),
        ).to(device)

    if use_text:
        msg = model.load_state_dict(
            torch.load(args.ckpt, map_location=device), strict=False
        )
    else:
        load_init_ckpt(model, args.ckpt)

    model.eval()

    post_pred = Compose([AsDiscrete(argmax=True)])
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    dices = []
    with torch.inference_mode():
        for i in range(len(ds)):
            b_any = ds[i]

            # If a train-time transform ever produced a list (shouldn't in eval), pick first:
            if isinstance(b_any, list):
                b_any = b_any[0]

            assert isinstance(b_any, dict), "Expected a dict from dataset in eval mode."
            b: Dict[str, Any] = cast(Dict[str, Any], b_any)

            x: torch.Tensor = cast(torch.Tensor, b["CTPT"]).unsqueeze(
                0
            )  # [1,C,D,H,W] on CPU
            y: torch.Tensor = cast(torch.Tensor, b["GT"]).unsqueeze(0).to(device)
            rid = str(b["ID"])

            tok = b.get("TXT", None)
            if tok is not None and not isinstance(tok, torch.Tensor):
                tok = torch.as_tensor(tok)
            txt_batch = tok.unsqueeze(0).to(device) if tok is not None else None

            roi = (args.infer_patch,) * 3

            def _predictor(z):
                with autocast(
                    "cuda",
                    enabled=(args.amp and device.type == "cuda"),
                    dtype=torch.float16,
                ):
                    out = model(z, txt_batch, None)[0]
                return out.float()

            sw = sliding_window_inference(
                x,
                roi_size=roi,
                sw_batch_size=1,
                predictor=_predictor,
                overlap=0.25,
                mode="gaussian",
                sw_device=device,
                device=torch.device("cpu"),
            )

            yhat_t: torch.Tensor = cast(torch.Tensor, post_pred(sw)[0])  # CPU tensor
            dice_metric(
                [F.one_hot(yhat_t.to(torch.long), num_classes=2).permute(3, 0, 1, 2)],
                [F.one_hot(y[0].to(torch.long), num_classes=2).permute(3, 0, 1, 2)],
            )
            agg = dice_metric.aggregate()
            dice_metric.reset()
            dice_t = agg[0] if isinstance(agg, tuple) else agg
            d = float(dice_t.item())
            dices.append(d)

            save_nifti_like(yhat_t.cpu().numpy(), outdir / f"{rid}.nii.gz")
            print(f"{rid}: Dice={d:.4f}")

            # ---- Optional attention heatmap at low-res (one extra pass) ----
            if args.save_heatmaps and txt_batch is not None:
                with torch.inference_mode(), autocast(
                    "cuda", enabled=(args.amp and device.type == "cuda")
                ):
                    # make a small, divisible-by-16 volume to keep memory trivial
                    D, H, W = x.shape[2:]
                    Sd = max(64, min(160, (D // 16) * 16))
                    Sh = max(64, min(160, (H // 16) * 16))
                    SwW = max(64, min(160, (W // 16) * 16))
                    import torch.nn.functional as F

                    xs = F.interpolate(
                        x.to(device),
                        size=(Sd, Sh, SwW),
                        mode="trilinear",
                        align_corners=False,
                    )
                    logits, attn = model(xs, txt_batch, None, return_attn=True)
                    if attn is not None:
                        # attn: (B, N, L) for last decoder level; sum over tokens → reshape → upsample
                        B, N, L = attn.shape
                        attn_sum = (
                            attn[0]
                            .sum(-1)
                            .reshape(Sd, Sh, SwW)
                            .unsqueeze(0)
                            .unsqueeze(0)
                        )  # [1,1,D,H,W]
                        attn_up = F.interpolate(
                            attn_sum,
                            size=(D, H, W),
                            mode="trilinear",
                            align_corners=False,
                        )
                        attn_np = attn_up[0, 0].clamp(min=0).cpu().numpy()
                        # normalize 0..255 for a quick-look NIfTI (uint8)
                        import numpy as np

                        a = attn_np - attn_np.min()
                        bmap = a / (a.max() + 1e-6)
                        save_nifti_like(
                            (bmap * 255).astype(np.uint8),
                            outdir / f"{rid}_attn_sum.nii.gz",
                        )

    import numpy as np

    print(f"Mean Dice (fold {args.fold}): {np.mean(dices):.4f}")
    with open(outdir / "dice.json", "w") as f:
        json.dump(
            {
                "fold": int(args.fold),
                "mean_dice": float(np.mean(dices)),
                "per_case": [float(x) for x in dices],
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
