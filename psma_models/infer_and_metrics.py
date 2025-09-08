"""
Runs inference on a manifest split, writes NIfTI predictions and optional attention heatmaps
projected to full resolution (upsampling from final decoder stage).
Also computes whole-body 3D Dice.

Usage:
python infer_and_metrics.py \
  --manifest data/dataset.csv \
  --fold 0 \
  --ckpt runs/exp1_text/models/fold0_ep0300.pt \
  --outdir runs/exp1_text \
  --text-modality image|radgraph|gpt_gpt_raw|msraw \
  --radgraph-root cache/text 
  --gpt-root cache/text
  --gpt-raw-root cache/text
  --msraw-root cache/text_msraw
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
from data.text_features import load_tokens_for_id

# from models.unet3d_text import UNet3DText
from typing import Dict, Any, cast
from nibabel.loadsave import save as nib_save
from nibabel.nifti1 import Nifti1Image
import pandas as pd
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from monai.networks.nets.segresnet import SegResNet
from models.segresnet_text import SegResNetText
from tools.ckpt_utils import load_init_ckpt
from tools.ckpt_utils import load_trained_ckpt

from torch.amp.autocast_mode import autocast


class _Wrap(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x, *_):
        return self.net(x), None


def save_nifti_like(array, out_path, affine):
    """
    Save uint8 array with an affine that may come as np.ndarray or torch.Tensor.
    """
    if isinstance(affine, torch.Tensor):
        affine = affine.cpu().numpy()
    nib_save(Nifti1Image(array.astype(np.uint8), affine), str(out_path))


def _safe_affine_from_meta(ctpt) -> np.ndarray:
    """
    Prefer MetaTensor.affine; fall back to meta['affine']; else identity.
    Returns a numpy 4x4.
    """
    aff = getattr(ctpt, "affine", None)
    if isinstance(aff, torch.Tensor):
        return aff.cpu().numpy()
    if aff is None and hasattr(ctpt, "meta"):
        aff2 = ctpt.meta.get("affine", None)
        if isinstance(aff2, torch.Tensor):
            return aff2.cpu().numpy()
        if aff2 is not None:
            return aff2
    return np.eye(4, dtype=np.float32)


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
        choices=["image", "radgraph", "gpt", "gpt_raw", "msraw"],
        default="image",
    )
    ap.add_argument("--radgraph-root", default="")
    ap.add_argument("--gpt-root", default="")
    ap.add_argument("--gpt-raw-root", default="")
    ap.add_argument("--msraw-root", default="")
    ap.add_argument("--infer_patch", type=int, default=192)
    ap.add_argument(
        "--split",
        choices=["val", "test"],
        default="test",
        help="val = CV validation fold; test = held-out test split from the manifest",
    )
    ap.add_argument(
        "--scores_csv",
        action="store_true",
        help="If set, write per-case Dice to <outdir>/<split>_fold<k>/dice_by_id.csv",
    )
    ap.add_argument("--save_heatmaps", action="store_true")
    ap.add_argument("--amp", action="store_true", help="Enable autocast (FP16)")
    args = ap.parse_args()

    base = Path(args.outdir) / f"{args.split}_fold{args.fold}"
    pred_dir = base / "preds"
    heat_dir = base / "heatmaps"
    pred_dir.mkdir(parents=True, exist_ok=True)
    if args.save_heatmaps:
        heat_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True  # ROI size is fixed, safe to enable

    df = pd.read_csv(args.manifest)
    if args.split == "test":
        rows = df[df["split"] == "test"].to_dict("records")
    else:  # "val"
        rows = df[(df["split"] != "test") & (df["fold"] == args.fold)].to_dict(
            "records"
        )

    t = make_transforms(
        train=False, include_mask=True, force_orient=False, pet_clip=35.0
    )
    text_roots = {
        "radgraph": Path(args.radgraph_root) if args.radgraph_root else Path("."),
        "gpt": Path(args.gpt_root) if args.gpt_root else Path("."),
        "gpt_raw": Path(args.gpt_raw_root) if args.gpt_raw_root else Path("."),
        "msraw": Path(args.msraw_root) if args.msraw_root else Path("."),
    }
    ds = PSMAJSONDataset(
        rows, t, text_modality=args.text_modality, text_roots=text_roots
    )

    txt_dim = infer_txt_dim_safely(
        args.text_modality, text_roots, rows, default_if_image=0
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
        base_model = SegResNet(
            spatial_dims=3,
            in_channels=2,
            out_channels=2,
            init_filters=16,
            blocks_down=(1, 2, 2, 4),
            blocks_up=(1, 1, 1),
        )
        model = _Wrap(base_model).to(device)

    load_trained_ckpt(model, args.ckpt, strict=True, verbose=True)

    model.eval()

    post_pred = Compose([AsDiscrete(argmax=True, to_onehot=2)])
    post_lab = Compose([AsDiscrete(to_onehot=2)])

    dice_metric = DiceMetric(include_background=False, reduction="mean")

    dices = []
    dice_by_id = []  # (ID, dice) for CSV
    ids_in_order = []  # keep the order that matches per_case
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
            ids_in_order.append(rid)

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

            # With batch size 1, index the batch dim directly and avoid decollate/list types.
            # sw: [1, C, D, H, W], y: [1, 1, D, H, W]
            yhat_oh = post_pred(sw[0])  # [2, D, H, W]
            gt_oh = post_lab(y[0].cpu())  # [2, D, H, W]

            assert (
                yhat_oh.shape == gt_oh.shape
            ), f"Bad shapes: pred {yhat_oh.shape}, gt {gt_oh.shape}"
            dice_metric([yhat_oh.float()], [gt_oh.float()])

            agg = dice_metric.aggregate()
            dice_metric.reset()
            dice_t = agg[0] if isinstance(agg, tuple) else agg
            d = float(dice_t.item())
            dices.append(d)
            dice_by_id.append({"ID": rid, "Dice": d})

            yhat_idx = torch.argmax(yhat_oh, dim=0)  # [D,H,W]
            aff = _safe_affine_from_meta(b["CTPT"])
            save_nifti_like(yhat_idx.cpu().numpy(), pred_dir / f"{rid}.nii.gz", aff)
            print(f"{rid}: Dice={d:.4f}")

            # ---- attention heatmap at low-res (one extra pass) ----
            if args.save_heatmaps and txt_batch is not None:
                with torch.inference_mode(), autocast(
                    "cuda", enabled=(args.amp and device.type == "cuda")
                ):
                    D, H, W = x.shape[2:]
                    Sd = max(64, min(160, (D // 16) * 16))
                    Sh = max(64, min(160, (H // 16) * 16))
                    Sw = max(64, min(160, (W // 16) * 16))

                    xs = F.interpolate(
                        x.to(device),
                        size=(Sd, Sh, Sw),
                        mode="trilinear",
                        align_corners=False,
                    )
                    logits, attn = model(xs, txt_batch, None, return_attn=True)
                    if attn is not None:
                        # attn could be (B, N, L) or (B, H, N, L); reduce to (B, N, L)
                        if attn.dim() == 4:
                            attn = attn.mean(dim=1)  # average over heads -> (B,N,L)
                        elif attn.dim() != 3:
                            raise RuntimeError(
                                f"Unexpected attn shape: {tuple(attn.shape)}"
                            )

                        N = int(attn.shape[1])  # number of query locations
                        # try the typical decoder scales (full, /2, /4, /8, /16)
                        shape = None
                        for f in (1, 2, 4, 8, 16):
                            if (Sd // f) * (Sh // f) * (Sw // f) == N:
                                shape = (Sd // f, Sh // f, Sw // f)
                                break
                        if shape is None:
                            raise RuntimeError(
                                f"Cannot map attention length N={N} to a grid from (Sd,Sh,Sw)=({Sd},{Sh},{Sw})."
                            )
                        # Sum over all query locations to get token importance (L,)
                        token_scores = attn[0].sum(dim=0).cpu().numpy()  # (L,)
                        tok_dir = base / "attn_tokens"
                        tok_dir.mkdir(parents=True, exist_ok=True)
                        np.save(tok_dir / f"{rid}_token_scores.npy", token_scores)

                        vol = (
                            attn[0].sum(-1).reshape(*shape).unsqueeze(0).unsqueeze(0)
                        )  # [1,1,*,*,*]
                        attn_up = F.interpolate(
                            vol, size=(D, H, W), mode="trilinear", align_corners=False
                        )
                        a = attn_up[0, 0].clamp(min=0).cpu().numpy()
                        bmap = (a - a.min()) / (a.max() + 1e-6)
                        save_nifti_like(
                            (bmap * 255).astype(np.uint8),
                            heat_dir / f"{rid}_attn_sum.nii.gz",
                            aff,
                        )

    mean_d = float(np.mean(dices)) if dices else 0.0
    print(
        f"Mean Dice ({args.split}{'' if args.split=='test' else f' fold {args.fold}'}): {mean_d:.4f}"
    )

    if args.scores_csv:
        pd.DataFrame(dice_by_id).to_csv(base / "dice_by_id.csv", index=False)

    with open(base / f"dice_{args.split}.json", "w") as f:
        json.dump(
            {
                "split": args.split,
                "fold": int(args.fold),
                "mean_dice": mean_d,
                "ids": ids_in_order,
                "per_case": [float(x) for x in dices],
                "dice_by_id": dice_by_id,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
