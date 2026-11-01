"""Evaluate a trained checkpoint on val or test with the full metric suite.

    python scripts/evaluate.py --run unet_a_smoke --split test
    python scripts/evaluate.py --run unet_a --split val --ckpt last

Writes results/<run>/eval_<split>_<ckpt>.json.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # must precede `import torch` on Windows

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from architecture.unet import UNet  # noqa: E402
from src.config import DataConfig, TrainConfig  # noqa: E402
from src.dataset import BrightDataset, build_loader  # noqa: E402
from src.engine import early_fusion_forward, evaluate  # noqa: E402
from src.utils import get_device, load_checkpoint  # noqa: E402

CLASS_NAMES = ["background", "intact", "damaged", "destroyed"]


def main():
    p = argparse.ArgumentParser(description="Evaluate a BRIGHT U-Net checkpoint")
    p.add_argument("--config", default=str(ROOT / "config" / "baseline.toml"))
    p.add_argument("--run", required=True, help="run name under checkpoints/")
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--ckpt", default="best", help="best | last | path to a .pth")
    p.add_argument("--num-workers", type=int)
    args = p.parse_args()

    cfg = TrainConfig.from_toml(args.config)
    dcfg = DataConfig.from_toml(args.config, args.split)
    num_workers = cfg.num_workers if args.num_workers is None else args.num_workers

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.suffix:
        ckpt_path = ROOT / "checkpoints" / args.run / f"{args.ckpt}.pth"
    device = get_device()

    model = UNet(in_channels=cfg.in_channels, num_classes=cfg.num_classes).to(device)
    state = load_checkpoint(ckpt_path, model, map_location=device)
    print(f"loaded {ckpt_path} (epoch {state.get('epoch')})")

    loader = build_loader(BrightDataset(dcfg), cfg.eval_batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=cfg.pin_memory)
    m = evaluate(model, loader, device, early_fusion_forward, cfg.num_classes,
                 cfg.eval_tile_size, desc=args.split)

    print(f"\n=== {args.run} | {args.split} ({len(loader.dataset)} tiles) | {ckpt_path.name} ===")
    print(f"mIoU (set-level) : {100 * m['miou']:.2f}")
    for n, v in zip(CLASS_NAMES, m["iou_per_class"]):
        print(f"  IoU {n:<11}: {100 * v:.2f}")
    print(f"F1_loc           : {100 * m['loc_f1']:.2f}")
    print(f"F1_clf (harmonic): {100 * m['clf_f1']:.2f}")
    for n, v in zip(CLASS_NAMES[1:], m["damage_f1_per_class"]):
        print(f"  F1 {n:<12}: {100 * v:.2f}")
    print(f"OA               : {100 * m['oa']:.2f}")

    out = ROOT / "results" / args.run / f"eval_{args.split}_{ckpt_path.stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"checkpoint": str(ckpt_path), "epoch": state.get("epoch"), "split": args.split,
                   "metrics": m}, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
