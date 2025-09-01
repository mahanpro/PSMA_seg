# psma_textfusion/train_ddp.py
"""
DDP training with 5-fold CV and four ablations.

Example (single node, 4 GPUs):
torchrun --nproc_per_node=4 psma_textfusion/train_ddp.py \
  --manifest data/dataset.csv \
  --fold 0 \
  --text-modality image|radgraph|gpt|msraw \
  --radgraph-root cache/text \
  --gpt-root      cache/text \
  --msraw-root    cache/text_msraw \
  --outdir runs/exp1_image_only \
  --epochs 300 --patch 128 --infer_patch 192 --lr 2e-4 --bs 1
"""
import argparse
from pathlib import Path
import torch, torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from monai.inferers.utils import sliding_window_inference
from monai.metrics.meandice import DiceMetric
from monai.transforms.compose import Compose
from monai.transforms.post.array import AsDiscrete
from monai.data.utils import decollate_batch
from models.unet3d_text import UNet3DText
from data.petct_dataset import PSMAJSONDataset, make_transforms
import pandas as pd
import monai

from textfusion.collate import text_list_data_collate

# from data.transforms_psma import make_transforms


def ddp_setup():
    dist.init_process_group(backend="nccl", init_method="env://")


def read_manifest(path: Path):
    return pd.read_csv(path)


def to_device(batch, device):
    out = {k: v for k, v in batch.items()}
    out["CTPT"] = out["CTPT"].to(device)
    if "GT" in out and out["GT"] is not None:
        out["GT"] = out["GT"].to(device)
    if out.get("TXT") is not None:
        out["TXT"] = out["TXT"].to(device)
    if out.get("TXT_MASK") is not None:
        out["TXT_MASK"] = out["TXT_MASK"].to(device)
    return out


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
    ap.add_argument("--patch", type=int, default=128)
    ap.add_argument("--infer_patch", type=int, default=192)
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--resume", type=str, default="")
    ap.add_argument("--max_tokens", type=int, default=256)
    args = ap.parse_args()

    ddp_setup()
    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    torch.cuda.set_device(device)

    outdir = Path(args.outdir)
    (outdir / "models").mkdir(parents=True, exist_ok=True)
    (outdir / "logs").mkdir(parents=True, exist_ok=True)

    df = read_manifest(Path(args.manifest))
    df_cv = df[df["split"] == "trainval"].copy() if "split" in df.columns else df
    train_rows = df_cv[df_cv["fold"] != args.fold].to_dict("records")
    val_rows = df_cv[df_cv["fold"] == args.fold].to_dict("records")

    train_transforms = make_transforms(
        train=True, include_mask=True, force_orient=False, pet_clip=20.0
    )
    valid_transforms = make_transforms(
        train=False, include_mask=True, force_orient=False, pet_clip=20.0
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
        collate_fn=text_list_data_collate,
    )
    val_ld = DataLoader(
        val_ds,
        batch_size=1,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=text_list_data_collate,
    )

    # infer text dim automatically
    txt_dim = 768
    for k in range(min(8, len(train_ds))):
        st = train_ds[k].get("TXT", None)
        if st is not None:
            txt_dim = st.shape[-1]
            break
    use_text = args.text_modality != "image"

    model = UNet3DText(
        in_ch=2, out_ch=2, base_ch=32, txt_dim=txt_dim, use_text=use_text, n_heads=4
    ).to(device)
    if args.resume:
        msg = model.load_state_dict(
            torch.load(args.resume, map_location=device), strict=False
        )
        if rank == 0:
            print("Loaded (non-strict):", msg)
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[device], find_unused_parameters=False
    )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=0.0
    )

    dice_metric = DiceMetric(include_background=False, reduction="mean")
    post_pred = Compose([AsDiscrete(argmax=True, to_onehot=2)])
    post_lab = Compose([AsDiscrete(to_onehot=2)])

    loss_fn = monai.losses.DiceCELoss(
        to_onehot_y=True, softmax=True, lambda_dice=1.0, lambda_ce=0.2
    )

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        running = 0.0
        steps = 0
        for batch in train_ld:
            batch = to_device(batch, device)
            if batch["TXT"] is None:
                logits, _ = model(batch["CTPT"], None, None, return_attn=False)
            else:
                logits, _ = model(
                    batch["CTPT"], batch["TXT"], batch["TXT_MASK"], return_attn=False
                )

            loss = loss_fn(logits, batch["GT"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += float(loss.detach().cpu())
            steps += 1

        sched.step()
        if rank == 0:
            print(
                f"[Fold {args.fold}] Epoch {epoch+1}/{args.epochs} | loss={running/max(1,steps):.4f}"
            )

        if (epoch + 1) % args.val_interval == 0:
            model.eval()
            dice_metric.reset()
            with torch.no_grad():
                for vb in val_ld:
                    vb = to_device(vb, device)
                    if vb["TXT"] is None:
                        pred, _ = model(vb["CTPT"], None, None, return_attn=False)
                        txt_b, msk_b = None, None
                    else:
                        pred, _ = model(
                            vb["CTPT"], vb["TXT"], vb["TXT_MASK"], return_attn=False
                        )
                        txt_b, msk_b = vb["TXT"], vb["TXT_MASK"]

                    roi = (args.infer_patch,) * 3
                    sw = sliding_window_inference(
                        vb["CTPT"],
                        roi,
                        sw_batch_size=1,
                        predictor=lambda x: model.module(x, txt_b, msk_b)[0],
                    )
                    y_hat = [post_pred(t) for t in decollate_batch(sw)]
                    y = [post_lab(t) for t in decollate_batch(vb["GT"])]
                    dice_metric(y_hat, y)

                d = dice_metric.aggregate().item()
                if rank == 0:
                    print(f"[Fold {args.fold}] VAL Dice={d:.4f} at epoch {epoch+1}")
                    ck = outdir / "models" / f"fold{args.fold}_ep{epoch+1:04d}.pth"
                    torch.save(model.module.state_dict(), ck)

    if rank == 0:
        print("Done.")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
