"""Train the variant-A U-Net baseline with periodic validation.

    python scripts/train.py --env local                                # smoke test (~3 min)
    python scripts/train.py --env cluster --run-name unet_a_s42        # full official schedule
    python scripts/train.py --env cluster --run-name unet_a_s42 --resume   # continue a run
    python scripts/train.py --env cluster --run-name unet_a_s43 --set train.seed=43

Config = config/baseline.toml -> config/env/<env>.toml -> $BRIGHT_DATA_ROOT -> --set
(see README "Configuration"). Writes checkpoints/<run>/{best,last}.pth and
results/<run>/{run_config.json,metrics.csv}.

Exit code 3 = stopped early by SIGTERM/SIGINT (e.g. SLURM time limit); last.pth holds
the last completed epoch, rerun with --resume to continue.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # must precede `import torch` on Windows

import argparse
import csv
import json
import math
import signal
import sys
import time
from dataclasses import asdict
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from architecture.unet import UNet  # noqa: E402
from src.config import DataConfig, TrainConfig, load_config  # noqa: E402
from src.dataset import BrightDataset, build_loader  # noqa: E402
from src.engine import AMP_DTYPES, early_fusion_forward, evaluate, train_one_epoch, use_channels_last  # noqa: E402
from src.losses import segmentation_loss  # noqa: E402
from src.utils import (get_device, git_info, load_checkpoint, machine_info, make_run_dirs,  # noqa: E402
                       rng_state, save_checkpoint, seed_worker, set_rng_state, set_seed)

CLASS_NAMES = ["background", "intact", "damaged", "destroyed"]
EXIT_STOPPED = 3

_STOP = False


def _request_stop(signum, _frame):
    global _STOP
    _STOP = True
    print(f"signal {signum} received: stopping after the current step, last.pth keeps the "
          f"last completed epoch", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description="Train the BRIGHT U-Net baseline (variant A)")
    p.add_argument("--config", default=str(ROOT / "config" / "baseline.toml"))
    p.add_argument("--env", help="machine profile: local | cluster | <name> | path/to.toml")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                   help="override any config key (repeatable), e.g. --set train.learning_rate=3e-4")
    p.add_argument("--resume", action="store_true", help="continue from checkpoints/<run>/last.pth if present")
    # shortcuts (kept from the original CLI); they are applied as --set overrides
    p.add_argument("--epochs", type=int, help="= --set train.epochs=N (also sets train.max_samples=0)")
    p.add_argument("--max-steps", type=int, help="= --set train.max_steps=N (steps per epoch cap)")
    p.add_argument("--run-name", help="= --set train.run_name=NAME")
    p.add_argument("--batch-size", type=int, help="= --set train.train_batch_size=N")
    p.add_argument("--num-workers", type=int, help="= --set runtime.num_workers=N")
    return p.parse_args()


def shortcut_overrides(args):
    o = []
    if args.epochs is not None:
        o += [f"train.epochs={args.epochs}", "train.max_samples=0"]
    if args.max_steps is not None:
        o.append(f"train.max_steps={args.max_steps}")
    if args.run_name is not None:
        o.append(f'train.run_name="{args.run_name}"')
    if args.batch_size is not None:
        o.append(f"train.train_batch_size={args.batch_size}")
    if args.num_workers is not None:
        o.append(f"runtime.num_workers={args.num_workers}")
    return o


def fmt_metrics(m):
    ious = " ".join(f"{n}={100 * v:.2f}" for n, v in zip(CLASS_NAMES, m["iou_per_class"]))
    return (f"mIoU={100 * m['miou']:.2f} event_mIoU={100 * m['event_miou']:.2f} | IoU: {ious} | "
            f"F1_loc={100 * m['loc_f1']:.2f} F1_clf={100 * m['clf_f1']:.2f} OA={100 * m['oa']:.2f}")


def main():
    args = parse_args()
    overrides = args.overrides + shortcut_overrides(args)
    raw = load_config(args.config, args.env, overrides)
    cfg = TrainConfig.from_dict(raw)
    train_dcfg = DataConfig.from_dict(raw, "train")
    val_dcfg = DataConfig.from_dict(raw, "val")

    set_seed(cfg.seed, cfg.deterministic)
    device = get_device()
    results_dir, ckpt_dir = make_run_dirs(cfg.run_name)
    last_path, best_path = ckpt_dir / "last.pth", ckpt_dir / "best.pth"
    if last_path.exists() and not args.resume:
        sys.exit(f"run {cfg.run_name!r} already has {last_path}; pass --resume to continue it, "
                 f"or choose a new --run-name")

    amp_dtype = AMP_DTYPES.get(cfg.amp) if device.type == "cuda" else None
    if amp_dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        sys.exit("runtime.amp='bf16' but this GPU has no bf16 support; use 'fp16' or 'none'")
    scaler = torch.amp.GradScaler("cuda") if amp_dtype is torch.float16 else None
    torch.backends.cuda.matmul.allow_tf32 = True  # cuDNN convs already use TF32 by default

    g = torch.Generator()
    g.manual_seed(cfg.seed)
    train_loader = build_loader(BrightDataset(train_dcfg), cfg.train_batch_size, shuffle=True,
                                num_workers=cfg.num_workers, drop_last=True, pin_memory=cfg.pin_memory,
                                worker_init_fn=seed_worker, generator=g)
    val_loader = build_loader(BrightDataset(val_dcfg), cfg.eval_batch_size, shuffle=False,
                              num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)

    steps_per_epoch = min(len(train_loader), cfg.max_steps) if cfg.max_steps else len(train_loader)
    samples_per_epoch = steps_per_epoch * cfg.train_batch_size
    epochs = math.ceil(cfg.max_samples / samples_per_epoch) if cfg.max_samples > 0 else cfg.epochs

    model = UNet(in_channels=cfg.in_channels, num_classes=cfg.num_classes).to(device)
    if cfg.channels_last and device.type == "cuda":
        use_channels_last(model)  # before the optimizer, so its state matches the layout
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    loss_fn = partial(segmentation_loss, ignore_index=train_dcfg.ignore_index,
                      lovasz_weight=cfg.lovasz_weight)

    start_epoch, best_miou, best_epoch = 1, -1.0, None
    if args.resume and last_path.exists():
        state = load_checkpoint(last_path, model, optimizer, map_location="cpu")
        extra = state.get("extra", {})
        if scaler is not None and extra.get("scaler"):
            scaler.load_state_dict(extra["scaler"])
        if extra.get("rng"):
            set_rng_state(extra["rng"])
        start_epoch = state["epoch"] + 1
        best_miou, best_epoch = state["best_metric"], extra.get("best_epoch")
        print(f"resumed {last_path} after epoch {state['epoch']} (best val mIoU {100 * best_miou:.2f})", flush=True)

    meta = {"config_file": str(Path(args.config).resolve()), "env_profile": args.env, "overrides": overrides,
            "train": asdict(cfg), "data_train": asdict(train_dcfg), "data_val": asdict(val_dcfg),
            "schedule": {"epochs": epochs, "steps_per_epoch": steps_per_epoch,
                         "total_steps": epochs * steps_per_epoch, "total_samples": epochs * samples_per_epoch},
            "resolved_config": raw, "git": git_info(), "machine": machine_info(device),
            "n_params": sum(p.numel() for p in model.parameters())}
    tag = "" if start_epoch == 1 else f"_resume_e{start_epoch}"
    with open(results_dir / f"run_config{tag}.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)

    print(f"device={device} ({meta['machine']['gpu'] or 'cpu'}) run={cfg.run_name} amp={cfg.amp} "
          f"channels_last={cfg.channels_last} deterministic={cfg.deterministic}", flush=True)
    print(f"train: {len(train_loader.dataset)} tiles, {steps_per_epoch} steps/epoch "
          f"(batch {cfg.train_batch_size}, crop {train_dcfg.crop_size}) x {epochs} epochs = "
          f"{epochs * steps_per_epoch} steps / {epochs * samples_per_epoch} samples | "
          f"val: {len(val_loader.dataset)} tiles (window {cfg.eval_tile_size or 'full'}, "
          f"every {cfg.val_every} epochs{f', first {cfg.eval_limit} tiles' if cfg.eval_limit else ''})", flush=True)
    if meta["git"]["dirty"]:
        print("WARNING: uncommitted changes to tracked files; this run is not tied to a commit", flush=True)
    elif meta["git"]["commit"] is None or meta["git"]["dirty"] is None:
        print("WARNING: git commit/state unknown (no git here?); this run is not tied to a commit", flush=True)

    csv_path = results_dir / "metrics.csv"
    fields = ["epoch", "train_loss", "val_miou", "val_event_miou"] + [f"val_iou_{n}" for n in CLASS_NAMES] + \
             ["val_loc_f1", "val_clf_f1", "val_oa", "epoch_time_s", "samples_per_s", "data_wait_frac"]
    if not csv_path.exists() or start_epoch == 1:
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(fields)

    signal.signal(signal.SIGTERM, _request_stop)  # SLURM sends SIGTERM before the time limit
    signal.signal(signal.SIGINT, _request_stop)
    quiet = not sys.stdout.isatty()  # no tqdm bars in SLURM log files

    for epoch in range(start_epoch, epochs + 1):
        t0, timing = time.time(), {}
        train_loss = train_one_epoch(model, train_loader, optimizer, device, loss_fn,
                                     early_fusion_forward, scaler=scaler,
                                     max_steps=cfg.max_steps or None,
                                     ignore_index=train_dcfg.ignore_index,
                                     desc=f"epoch {epoch}/{epochs}", amp_dtype=amp_dtype,
                                     should_stop=lambda: _STOP, disable_pbar=quiet, timing=timing)
        if train_loss is None:  # stopped mid-epoch: last.pth still holds epoch-1
            print(f"stopped during epoch {epoch}; rerun with --resume to continue", flush=True)
            sys.exit(EXIT_STOPPED)

        row = {"epoch": epoch, "train_loss": f"{train_loss:.5f}",
               "samples_per_s": f"{timing['steps'] * cfg.train_batch_size / timing['seconds']:.1f}",
               "data_wait_frac": f"{timing['data_wait'] / timing['seconds']:.3f}"}
        msg = (f"[epoch {epoch}/{epochs}] train_loss={train_loss:.4f} "
               f"{row['samples_per_s']} samples/s data-wait {100 * float(row['data_wait_frac']):.0f}%")

        if epoch % cfg.val_every == 0 or epoch == epochs:
            m = evaluate(model, val_loader, device, early_fusion_forward, cfg.num_classes,
                         cfg.eval_tile_size, desc="val", limit=cfg.eval_limit, disable_pbar=quiet)
            row.update({"val_miou": f"{m['miou']:.5f}", "val_event_miou": f"{m['event_miou']:.5f}",
                        "val_loc_f1": f"{m['loc_f1']:.5f}", "val_clf_f1": f"{m['clf_f1']:.5f}",
                        "val_oa": f"{m['oa']:.5f}"})
            row.update({f"val_iou_{n}": f"{v:.5f}" for n, v in zip(CLASS_NAMES, m["iou_per_class"])})
            msg += f" | val {fmt_metrics(m)}"
            if m["miou"] > best_miou:
                best_miou, best_epoch = m["miou"], epoch
                save_checkpoint(best_path, model, optimizer, epoch, best_miou,
                                {"val_metrics": m, "train": asdict(cfg)})
                msg += "  * new best"

        save_checkpoint(last_path, model, optimizer, epoch, best_miou,
                        {"train": asdict(cfg), "rng": rng_state(), "best_epoch": best_epoch,
                         "scaler": scaler.state_dict() if scaler is not None else None})
        row["epoch_time_s"] = f"{time.time() - t0:.1f}"
        with open(csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writerow(row)
        eta_h = (epochs - epoch) * (time.time() - t0) / 3600
        print(msg + f"  ({row['epoch_time_s']}s, ETA {eta_h:.1f} h)", flush=True)

        if _STOP:
            print(f"stopped after epoch {epoch}; rerun with --resume to continue", flush=True)
            sys.exit(EXIT_STOPPED)

    print(f"done. best val mIoU={100 * best_miou:.2f} at epoch {best_epoch} -> {best_path}", flush=True)


if __name__ == "__main__":
    main()
