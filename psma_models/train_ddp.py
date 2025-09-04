"""
DDP training with 5-fold CV and four ablations.

Example (single node, 1 GPU):
torchrun --nproc_per_node=1 train_ddp.py \
  --manifest data/dataset.csv \
  --fold 0 \
  --init-ckpt /weights/segresnet_fold0.pth \
  --text-modality image|radgraph|gpt|msraw \
  --radgraph-root cache/text \
  --gpt-root      cache/text \
  --msraw-root    cache/text_msraw \
  --outdir runs/exp1_image_only \
  --epochs 300 --patch 128 --infer_patch 192 --lr 2e-4 --bs 2
"""

import argparse, os
from pathlib import Path
import json
import torch, torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from monai.inferers.utils import sliding_window_inference
from monai.metrics.meandice import DiceMetric
from monai.transforms.compose import Compose
from monai.transforms.post.array import AsDiscrete
from monai.data.utils import decollate_batch
from data.text_features import load_tokens_for_id
from typing import List, cast

# from models.unet3d_text import UNet3DText
from data.petct_dataset import PSMAJSONDataset, make_transforms
import pandas as pd
import matplotlib.pyplot as plt
import monai
from monai.losses.dice import DiceCELoss
from models.segresnet_text import SegResNetText

import torch.backends.cudnn as cudnn

# from torch.cuda.amp import GradScaler
from textfusion.collate import text_list_data_collate
from tools.ckpt_utils import load_init_ckpt
from tools.ckpt_utils import port_monai_to_segresnet_text

# from monai.networks.nets.unet import UNet as MonaiUNet
from monai.networks.nets.segresnet import SegResNet

# from data.transforms_psma import make_transforms


def ddp_setup():
    dist.init_process_group(backend="nccl", init_method="env://")


def read_manifest(path: Path):
    return pd.read_csv(path)


def to_device(batch, device):
    out = {k: v for k, v in batch.items()}
    out["CTPT"] = out["CTPT"].to(device, non_blocking=True)
    if "GT" in out and out["GT"] is not None:
        out["GT"] = out["GT"].to(device, non_blocking=True)
    if out.get("TXT") is not None:
        out["TXT"] = out["TXT"].to(device, non_blocking=True)
    if out.get("TXT_MASK") is not None:
        out["TXT_MASK"] = out["TXT_MASK"].to(device, non_blocking=True)
    return out


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


# small adapter so non-text models match (logits, attn) API
class WrapPlainNet(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x, *_args, **_kwargs):
        return self.net(x), None


def build_model(arch, use_text, txt_dim):
    if arch != "segresnet":
        raise ValueError("Only 'segresnet' is supported now.")
    if use_text:
        return SegResNetText(
            in_channels=2,
            out_channels=2,
            init_filters=16,
            blocks_down=[1, 2, 2, 4],
            blocks_up=[1, 1, 1],
            txt_dim=txt_dim,
            use_text=True,
            n_heads=4,
        )
    else:
        # plain MONAI SegResNet as the image-only baseline
        m = SegResNet(
            spatial_dims=3,
            in_channels=2,
            out_channels=2,
            init_filters=16,
            blocks_down=(1, 2, 2, 4),
            blocks_up=(1, 1, 1),
        )
        return WrapPlainNet(m)


def maybe_set_benchmark(fixed_patch: int | None):
    cudnn.benchmark = bool(fixed_patch and fixed_patch > 0)


def is_rank0():
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def save_ckpt(path, epoch, model, optimizer, scheduler, scaler, best_metric):
    state = {
        "epoch": epoch,
        "model": (
            model.module.state_dict()
            if hasattr(model, "module")
            else model.state_dict()
        ),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "best_metric": best_metric,
    }
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)  # atomic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument(
        "--text-modality",
        choices=["image", "radgraph", "gpt", "msraw"],
        default="image",
    )
    ap.add_argument("--radgraph-root", default="")
    ap.add_argument("--gpt-root", default="")
    ap.add_argument("--msraw-root", default="")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--val-interval", type=int, default=4)
    ap.add_argument(
        "--patch", type=int, default=128
    )  # FIXED patch is actually used below
    ap.add_argument("--infer_patch", type=int, default=192)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--max_tokens", type=int, default=256)

    # new
    ap.add_argument(
        "--arch",
        default="segresnet",
        choices=["segresnet"],
    )
    ap.add_argument("--init-ckpt", default="", help="Optional init checkpoint.")
    ap.add_argument("--amp", action="store_true", default=True)

    args = ap.parse_args()

    ddp_setup()
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    maybe_set_benchmark(args.patch)

    outdir = Path(args.outdir)
    logs_dir = outdir / "logs"
    best_path = str(outdir / "best.pt")
    last_path = str(outdir / "last.pt")

    logs_dir.mkdir(parents=True, exist_ok=True)

    # experiment "tag" = folder name of outdir
    exp_name = outdir.name

    # echo the tag so it shows up in logs
    if rank == 0:
        print(f"[EXP] name={exp_name} | fold={args.fold}")

    df = read_manifest(Path(args.manifest))
    df_cv = df[df["split"] == "trainval"].copy() if "split" in df.columns else df
    train_rows = df_cv[df_cv["fold"] != args.fold].to_dict("records")
    val_rows = df_cv[df_cv["fold"] == args.fold].to_dict("records")

    train_transforms = make_transforms(
        train=True,
        include_mask=True,
        force_orient=False,
        pet_clip=35.0,
        patch=args.patch,
    )
    valid_transforms = make_transforms(
        train=False, include_mask=True, force_orient=False, pet_clip=35.0, patch=None
    )

    from pathlib import Path as _P

    text_roots = {
        "radgraph": _P(args.radgraph_root) if args.radgraph_root else _P("."),
        "gpt": _P(args.gpt_root) if args.gpt_root else _P("."),
        "msraw": _P(args.msraw_root) if args.msraw_root else _P("."),
    }

    train_ds = PSMAJSONDataset(
        train_rows,
        train_transforms,
        text_modality=args.text_modality,
        text_roots=text_roots,
        max_tokens=args.max_tokens,
    )
    val_ds = PSMAJSONDataset(
        val_rows,
        valid_transforms,
        text_modality=args.text_modality,
        text_roots=text_roots,
        max_tokens=args.max_tokens,
    )

    train_sampler = DistributedSampler(train_ds, shuffle=True)
    val_sampler = DistributedSampler(val_ds, shuffle=False)

    train_ld = DataLoader(
        train_ds,
        batch_size=args.bs,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=text_list_data_collate,
    )
    val_ld = DataLoader(
        val_ds,
        batch_size=1,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=text_list_data_collate,
    )

    txt_dim = infer_txt_dim_safely(
        args.text_modality, text_roots, train_rows, default_if_image=0
    )
    use_text = args.text_modality != "image"

    model = build_model(args.arch, use_text, txt_dim).to(device)

    if args.init_ckpt:
        if use_text:
            state_dict = torch.load(args.init_ckpt, map_location="cpu")
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            port_monai_to_segresnet_text(model, state_dict, verbose=is_rank0())
        else:
            load_init_ckpt(model, args.init_ckpt, verbose=is_rank0())

    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    # ---- Warmup (linear) → Cosine ----
    warm_epochs = min(max(1, args.epochs // 10), max(1, args.epochs - 1))  # ~10% of run
    warm = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, total_iters=warm_epochs
    )
    cos = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs - warm_epochs), eta_min=0.0
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warm, cos], milestones=[warm_epochs]
    )
    if rank == 0:
        print(f"LR warmup epochs: {warm_epochs}  |  base LR: {args.lr}")

    dice_metric = DiceMetric(include_background=False, reduction="mean")
    post_pred = Compose([AsDiscrete(argmax=True, to_onehot=2)])
    post_lab = Compose([AsDiscrete(to_onehot=2)])
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True, lambda_dice=1.0, lambda_ce=0.2)

    # logs
    tr_losses, va_losses, va_dices, ep_nums = [], [], [], []

    from torch.cuda.amp import GradScaler

    scaler = GradScaler(enabled=args.amp)

    best_metric = -1.0
    best_epoch = 0

    for epoch in range(args.epochs):
        # ------------ train ------------
        train_sampler.set_epoch(epoch)
        model.train()
        running, steps = 0.0, 0
        for batch in train_ld:
            batch = to_device(batch, device)
            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=args.amp
            ):
                logits, _ = model(
                    batch["CTPT"], batch.get("TXT"), batch.get("TXT_MASK")
                )
                loss = loss_fn(logits, batch["GT"])
            optimizer.zero_grad(set_to_none=True)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.detach().cpu())
            steps += 1

        tr_loss = running / max(1, steps)
        scheduler.step()
        if rank == 0:
            print(
                f"[Fold {args.fold}] Epoch {epoch+1}/{args.epochs} | loss={tr_loss:.4f}"
            )

        dice_metric.reset()
        val_loss_sum = 0.0  # sum of per-item losses on this rank
        val_items = 0  # number of validation items on this rank
        # ------------ validate ------------
        if (epoch + 1) % args.val_interval == 0:
            model.eval()
            val_running, val_steps = 0.0, 0
            with torch.inference_mode():
                for vb in val_ld:
                    vb = to_device(vb, device)  # small batches, OK to move fully
                    B = vb["CTPT"].shape[0]
                    roi = (args.infer_patch,) * 3

                    def _predictor(z):
                        with torch.autocast(
                            device_type="cuda", enabled=args.amp, dtype=torch.float16
                        ):
                            out = model(z, vb.get("TXT"), vb.get("TXT_MASK"))[0]
                        return out.float()

                    sw = sliding_window_inference(
                        vb["CTPT"],
                        roi,
                        sw_batch_size=1,
                        predictor=_predictor,
                        overlap=0.25,
                        mode="gaussian",
                        sw_device=device,
                        device=device,
                    )
                    # accumulate loss as a SUM over items
                    loss_b = loss_fn(sw, vb["GT"])  # batch-mean
                    val_loss_sum += float(loss_b) * B
                    val_items += B
                    # Dice
                    y_hat: List[torch.Tensor] = cast(
                        List[torch.Tensor], [post_pred(t) for t in decollate_batch(sw)]
                    )
                    y: List[torch.Tensor] = cast(
                        List[torch.Tensor],
                        [post_lab(t) for t in decollate_batch(vb["GT"])],
                    )
                    dice_metric(y_hat, y)

            agg = dice_metric.aggregate()
            dice_t = agg[0] if isinstance(agg, (list, tuple)) else agg
            local_dice_mean = float(dice_t.item()) if val_items > 0 else 0.0

            dice_metric.reset()

            loss_sum = torch.tensor([val_loss_sum], device=device)
            cnt = torch.tensor([val_items], device=device)
            dice_sum = torch.tensor([local_dice_mean * val_items], device=device)

            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(cnt, op=dist.ReduceOp.SUM)
                dist.all_reduce(dice_sum, op=dist.ReduceOp.SUM)

            val_loss = (loss_sum / torch.clamp(cnt, min=1)).item()
            val_dice = (dice_sum / torch.clamp(cnt, min=1)).item()

            # update "best"
            if is_rank0() and val_dice > best_metric + 1e-6:
                best_metric = val_dice
                best_epoch = epoch + 1
                save_ckpt(
                    best_path,
                    epoch + 1,
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    best_metric,
                )

            if rank == 0:
                print(
                    f"[Fold {args.fold}] VAL loss={val_loss:.4f} Dice={val_dice:.4f} at epoch {epoch+1}"
                )

                # metrics log -> CSV + jpg
                ep_nums.append(epoch + 1)
                tr_losses.append(tr_loss)
                va_losses.append(val_loss)
                va_dices.append(val_dice)
                log_csv = logs_dir / f"metrics_{exp_name}_fold{args.fold}.csv"
                pd.DataFrame(
                    {
                        "epoch": ep_nums,
                        "train_loss": tr_losses,
                        "val_loss": va_losses,
                        "val_dice": va_dices,
                    }
                ).to_csv(log_csv, index=False)

                # Save curves
                fig = plt.figure(figsize=(8, 4))
                ax1 = fig.add_subplot(1, 2, 1)
                ax1.plot(ep_nums, tr_losses, label="train loss")
                ax1.plot(ep_nums, va_losses, label="val loss")
                ax1.set_title("Loss")
                ax1.set_xlabel("epoch")
                ax1.legend()
                ax2 = fig.add_subplot(1, 2, 2)
                ax2.plot(ep_nums, va_dices, label="val dice")
                ax2.set_title("Val Dice")
                ax2.set_xlabel("epoch")
                ax2.legend()
                fig.tight_layout()
                curves_path = logs_dir / f"curves_{exp_name}_fold{args.fold}.jpg"
                plt.savefig(curves_path, dpi=150)
                plt.close(fig)

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    if is_rank0():
        save_ckpt(last_path, epoch, model, optimizer, scheduler, scaler, best_metric)

    if rank == 0:
        print("Done.")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
