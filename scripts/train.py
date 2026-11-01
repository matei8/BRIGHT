"""Train the variant-A U-Net baseline with per-epoch validation.

    python scripts/train.py                                  # 2-epoch smoke (config default)
    python scripts/train.py --epochs 50 --run-name unet_a    # real run

Writes checkpoints/<run>/{best,last}.pth and results/<run>/{run_config.json,metrics.csv}.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # must precede `import torch` on Windows

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict, replace
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from architecture.unet import UNet  # noqa: E402
from src.config import DataConfig, TrainConfig  # noqa: E402
from src.dataset import BrightDataset, build_loader  # noqa: E402
from src.engine import early_fusion_forward, evaluate, train_one_epoch  # noqa: E402
from src.losses import segmentation_loss  # noqa: E402
from src.utils import get_device, make_run_dirs, save_checkpoint, seed_worker, set_seed  # noqa: E402

CLASS_NAMES = ["background", "intact", "damaged", "destroyed"]


def parse_args():
    p = argparse.ArgumentParser(description="Train the BRIGHT U-Net baseline (variant A)")
    p.add_argument("--config", default=str(ROOT / "config" / "baseline.toml"))
    p.add_argument("--epochs", type=int, help="override [train].epochs")
    p.add_argument("--max-steps", type=int, help="override [train].max_steps (steps per epoch cap)")
    p.add_argument("--run-name", help="override [train].run_name")
    p.add_argument("--batch-size", type=int, help="override [train].train_batch_size")
    p.add_argument("--num-workers", type=int, help="override [runtime].num_workers")
    return p.parse_args()


def fmt_metrics(m):
    ious = " ".join(f"{n}={100 * v:.2f}" for n, v in zip(CLASS_NAMES, m["iou_per_class"]))
    return (f"mIoU={100 * m['miou']:.2f} | IoU: {ious} | F1_loc={100 * m['loc_f1']:.2f} "
            f"F1_clf={100 * m['clf_f1']:.2f} OA={100 * m['oa']:.2f}")


def main():
    args = parse_args()
    cfg = TrainConfig.from_toml(args.config)
    overrides = {
        "epochs": args.epochs, "max_steps": args.max_steps, "run_name": args.run_name,
        "train_batch_size": args.batch_size, "num_workers": args.num_workers,
    }
    cfg = replace(cfg, **{k: v for k, v in overrides.items() if v is not None})
    train_dcfg = DataConfig.from_toml(args.config, "train")
    val_dcfg = DataConfig.from_toml(args.config, "val")

    set_seed(cfg.seed, cfg.deterministic)
    device = get_device()
    results_dir, ckpt_dir = make_run_dirs(cfg.run_name)
    print(f"device={device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'}) "
          f"run={cfg.run_name}")

    with open(results_dir / "run_config.json", "w") as f:
        json.dump({"config_file": str(Path(args.config).resolve()), "train": asdict(cfg),
                   "data_train": asdict(train_dcfg), "data_val": asdict(val_dcfg)}, f, indent=2)

    g = torch.Generator()
    g.manual_seed(cfg.seed)
    train_loader = build_loader(BrightDataset(train_dcfg), cfg.train_batch_size, shuffle=True,
                                num_workers=cfg.num_workers, drop_last=True, pin_memory=cfg.pin_memory,
                                worker_init_fn=seed_worker, generator=g)
    val_loader = build_loader(BrightDataset(val_dcfg), cfg.eval_batch_size, shuffle=False,
                              num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)
    print(f"train: {len(train_loader.dataset)} tiles, {len(train_loader)} steps/epoch "
          f"(batch {cfg.train_batch_size}, crop {train_dcfg.crop_size}) | "
          f"val: {len(val_loader.dataset)} tiles (window {cfg.eval_tile_size})")

    model = UNet(in_channels=cfg.in_channels, num_classes=cfg.num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    use_amp = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    loss_fn = partial(segmentation_loss, ignore_index=train_dcfg.ignore_index,
                      lovasz_weight=cfg.lovasz_weight)

    csv_path = results_dir / "metrics.csv"
    fields = ["epoch", "train_loss", "val_miou"] + [f"val_iou_{n}" for n in CLASS_NAMES] + \
             ["val_loc_f1", "val_clf_f1", "val_oa", "epoch_time_s"]
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(fields)

    best_miou = -1.0
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device, loss_fn,
                                     early_fusion_forward, scaler=scaler,
                                     max_steps=cfg.max_steps or None,
                                     ignore_index=train_dcfg.ignore_index,
                                     desc=f"epoch {epoch}/{cfg.epochs}")
        row = {"epoch": epoch, "train_loss": f"{train_loss:.5f}"}

        msg = f"[epoch {epoch}/{cfg.epochs}] train_loss={train_loss:.4f}"
        if epoch % cfg.val_every == 0 or epoch == cfg.epochs:
            m = evaluate(model, val_loader, device, early_fusion_forward, cfg.num_classes,
                         cfg.eval_tile_size, desc="val")
            row.update({"val_miou": f"{m['miou']:.5f}", "val_loc_f1": f"{m['loc_f1']:.5f}",
                        "val_clf_f1": f"{m['clf_f1']:.5f}", "val_oa": f"{m['oa']:.5f}"})
            row.update({f"val_iou_{n}": f"{v:.5f}" for n, v in zip(CLASS_NAMES, m["iou_per_class"])})
            msg += f" | val {fmt_metrics(m)}"
            if m["miou"] > best_miou:
                best_miou = m["miou"]
                save_checkpoint(ckpt_dir / "best.pth", model, optimizer, epoch, best_miou,
                                {"val_metrics": m, "train": asdict(cfg)})
                msg += "  * new best"

        save_checkpoint(ckpt_dir / "last.pth", model, optimizer, epoch, best_miou, {"train": asdict(cfg)})
        row["epoch_time_s"] = f"{time.time() - t0:.1f}"
        with open(csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writerow(row)
        print(msg + f"  ({row['epoch_time_s']}s)", flush=True)

    print(f"done. best val mIoU={100 * best_miou:.2f} -> {ckpt_dir / 'best.pth'}")


if __name__ == "__main__":
    main()
