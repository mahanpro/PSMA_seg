# psma_textfusion/infer_and_metrics.py
"""
Runs inference on a manifest split, writes NIfTI predictions and optional attention heatmaps
projected to full resolution (upsampling from final decoder stage).
Also computes whole-body 3D Dice.

Usage:
python psma_textfusion/infer_and_metrics.py \
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
from models.unet3d_text import UNet3DText
import pandas as pd
import torch.nn.functional as F


def read_manifest(path: Path):
    return pd.read_csv(path)


def save_nifti_like(array, out_path):
    nib.save(nib.Nifti1Image(array.astype(np.uint8), affine=np.eye(4)), str(out_path))


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
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(args.manifest)
    val_rows = df[df["fold"] == args.fold].to_dict("records")

    t = make_transforms(
        train=False, include_mask=True, force_orient=False, pet_clip=20.0
    )
    text_roots = {
        "radgraph": Path(args.radgraph_root) if args.radgraph_root else Path("."),
        "gpt": Path(args.gpt_root) if args.gpt_root else Path("."),
        "msraw": Path(args.msraw_root) if args.msraw_root else Path("."),
    }
    ds = PSMAJSONDataset(
        val_rows, t, text_modality=args.text_modality, text_roots=text_roots
    )
    post_pred = Compose([AsDiscrete(argmax=True)])
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    # infer text dim
    txt_dim = 768
    for k in range(min(8, len(ds))):
        st = ds[k].get("TXT", None)
        if st is not None:
            txt_dim = st.shape[-1]
            break
    use_text = args.text_modality != "image"

    model = UNet3DText(
        in_ch=2, out_ch=2, base_ch=32, txt_dim=txt_dim, use_text=use_text
    ).to(device)
    msg = model.load_state_dict(
        torch.load(args.ckpt, map_location=device), strict=False
    )
    print("Loaded:", msg)
    model.eval()

    post_pred = Compose([AsDiscrete(argmax=True)])
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    dices = []
    for i in range(len(ds)):
        b = ds[i]
        x = b["CTPT"].unsqueeze(0).to(device)
        y = b["GT"].unsqueeze(0).to(device)
        rid = b["ID"]
        tok, msk = b.get("TXT", None), None  # no padding when single item → no mask
        txt_batch = tok.unsqueeze(0).to(device) if tok is not None else None

        roi = (args.infer_patch,) * 3
        if txt_batch is None:
            sw = sliding_window_inference(
                x, roi, 1, predictor=lambda z: model(z, None, None)[0]
            )
        else:
            sw = sliding_window_inference(
                x, roi, 1, predictor=lambda z: model(z, txt_batch, None)[0]
            )

        yhat = post_pred(sw)[0]  # (D,H,W)
        # one-hot for metric
        dice_metric(
            [F.one_hot(yhat.long(), num_classes=2).permute(3, 0, 1, 2)],
            [F.one_hot(y[0].long(), num_classes=2).permute(3, 0, 1, 2)],
        )
        d = dice_metric.aggregate().item()
        dice_metric.reset()
        dices.append(d)
        save_nifti_like(yhat.cpu().numpy(), outdir / f"{rid}.nii.gz")

        if args.save_heatmaps and use_text and txt_batch is not None:
            with torch.no_grad():
                logits, attn = model(x, txt_batch, None, return_attn=True)
            if attn is not None:
                # attn: (B, Nq, L) → sum over L → (Nq,) → reshape to d1 spatial → upsample
                d1_sp = logits.shape[2:]  # (D',H',W')
                heat = attn[0].sum(dim=2)  # (Nq,)
                heat3d = heat.view(*d1_sp)  # (D',H',W')
                heat_up = F.interpolate(
                    heat3d[None, None],
                    size=x.shape[2:],
                    mode="trilinear",
                    align_corners=False,
                )[0, 0]
                heat_up = (heat_up - heat_up.min()) / (
                    heat_up.max() - heat_up.min() + 1e-6
                )
                np_heat = (heat_up.cpu().numpy() * 255).astype(np.uint8)
                save_nifti_like(np_heat, outdir / f"{rid}_attn.nii.gz")

        print(f"{rid}: Dice={d:.4f}")

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
