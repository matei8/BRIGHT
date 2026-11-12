"""Find the training bottleneck: data loading vs GPU compute, measured separately.

    python scripts/benchmark.py --env cluster                  # both parts
    python scripts/benchmark.py --env cluster --part gpu       # GPU only (synthetic data)
    python scripts/benchmark.py --env cluster --part loader --workers 8 16 32

* loader: real BrightDataset + augmentations, no GPU. samples/s the input pipeline can
  deliver with N workers.
* gpu:    synthetic batches already on the GPU, the real model + loss + AdamW step.
  samples/s the GPU can train at, for each speed option (cuDNN determinism,
  channels_last memory format, torch.compile).

Training runs at min(loader, gpu). Whichever is lower is the bottleneck. Nothing here
changes the training recipe (batch, crop, lr, loss stay identical).
"""

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch  # noqa: E402

from architecture.unet import UNet  # noqa: E402
from src.config import DataConfig, TrainConfig, load_config  # noqa: E402
from src.dataset import BrightDataset, build_loader  # noqa: E402
from src.engine import AMP_DTYPES  # noqa: E402
from src.losses import segmentation_loss  # noqa: E402
from src.utils import seed_worker  # noqa: E402


def bench_loader(raw, cfg, workers_list, n_batches):
    ds = BrightDataset(DataConfig.from_dict(raw, "train"))
    print(f"\n== data loader (batch {cfg.train_batch_size}, crop {ds.cfg.crop_size}, {os.cpu_count()} CPUs visible, "
          f"{len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else '?'} usable)")
    for nw in workers_list:
        loader = build_loader(ds, cfg.train_batch_size, shuffle=True, num_workers=nw, drop_last=True,
                              pin_memory=False, worker_init_fn=seed_worker)
        it = iter(loader)
        warm = min(max(nw, 2), n_batches // 2)  # let every worker deliver once
        for _ in range(warm):
            next(it)
        t0 = time.time()
        for _ in range(n_batches):
            next(it)
        dt = time.time() - t0
        print(f"  workers={nw:3d}: {n_batches * cfg.train_batch_size / dt:7.1f} samples/s", flush=True)
        del it, loader


def bench_gpu(cfg, crop, steps, variants):
    dev = torch.device("cuda")
    B = cfg.train_batch_size
    amp_dtype = AMP_DTYPES.get(cfg.amp)
    g = torch.Generator(device=dev).manual_seed(0)
    optical = torch.randn(B, 3, crop, crop, device=dev, generator=g)
    sar = torch.randn(B, 1, crop, crop, device=dev, generator=g)
    # realistic class mix (~86% background) so the Lovasz sort sees typical data
    labels = torch.multinomial(torch.tensor([0.86, 0.11, 0.02, 0.01], device=dev), B * crop * crop,
                               replacement=True, generator=g).view(B, crop, crop)
    print(f"\n== GPU step, synthetic data ({torch.cuda.get_device_name(0)}, batch {B}, crop {crop}, amp {cfg.amp})")

    for name, det, cl, comp in variants:
        torch.backends.cudnn.deterministic = det
        torch.backends.cudnn.benchmark = not det
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.manual_seed(0)
        model = UNet(cfg.in_channels, cfg.num_classes).to(dev)
        fmt = torch.channels_last if cl else torch.contiguous_format
        model = model.to(memory_format=fmt)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        fwd = model
        if comp:
            try:
                fwd = torch.compile(model)
            except Exception as e:  # noqa: BLE001
                print(f"  {name:32s} torch.compile unavailable: {type(e).__name__}: {e}")
                continue
        x = torch.cat([optical, sar], 1).contiguous(memory_format=fmt)

        def step():
            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                logits = fwd(x)
            loss = segmentation_loss(logits, labels, lovasz_weight=cfg.lovasz_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        try:
            torch.cuda.reset_peak_memory_stats()
            for _ in range(10 if comp else 5):  # warm-up (compile + cudnn autotune)
                step()
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(steps):
                step()
            torch.cuda.synchronize()
            dt = time.time() - t0
        except Exception as e:  # noqa: BLE001
            print(f"  {name:32s} failed: {type(e).__name__}: {str(e).splitlines()[0][:120]}")
            continue

        # share of the step spent in the loss (CE + Lovasz), timed separately
        with torch.no_grad(), torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = fwd(x)
        torch.cuda.synchronize()
        t1 = time.time()
        for _ in range(10):
            segmentation_loss(logits, labels, lovasz_weight=cfg.lovasz_weight)
        torch.cuda.synchronize()
        loss_ms = (time.time() - t1) / 10 * 1000

        sps = steps * B / dt
        tflops = sps * 1.81 * (crop / 640) ** 2  # measured: 1.81 TFLOP per 640^2 sample (fwd+bwd)
        print(f"  {name:32s} {sps:7.1f} samples/s  {dt / steps * 1000:6.0f} ms/step  "
              f"(loss {loss_ms:4.0f} ms)  ~{tflops:5.0f} TFLOPS  peak mem {torch.cuda.max_memory_allocated() / 2**30:5.1f} GB",
              flush=True)
        del model, opt, fwd
        torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser(description="Benchmark data loading vs GPU compute")
    p.add_argument("--config", default=str(ROOT / "config" / "baseline.toml"))
    p.add_argument("--env")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--part", choices=["all", "loader", "gpu"], default="all")
    p.add_argument("--workers", type=int, nargs="+", help="loader worker counts to try (default: 8, 16, all CPUs)")
    p.add_argument("--batches", type=int, default=40, help="loader batches to time")
    p.add_argument("--steps", type=int, default=30, help="GPU steps to time")
    p.add_argument("--no-compile", action="store_true", help="skip the torch.compile variant")
    args = p.parse_args()

    raw = load_config(args.config, args.env, args.overrides)
    cfg = TrainConfig.from_dict(raw)
    crop = int(raw["data"]["crop_size"])

    if args.part in ("all", "gpu"):
        if not torch.cuda.is_available():
            sys.exit("no GPU visible (forgot --nv?)")
        variants = [  # name, cudnn deterministic, channels_last, torch.compile
            ("current (deterministic)", True, False, False),
            ("deterministic=false", False, False, False),
            ("+ channels_last", False, True, False),
        ]
        if not args.no_compile:
            variants.append(("+ channels_last + torch.compile", False, True, True))
        bench_gpu(cfg, crop, args.steps, variants)

    if args.part in ("all", "loader"):
        ncpu = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
        workers = args.workers or sorted({8, 16, ncpu})
        bench_loader(raw, cfg, workers, args.batches)

    print("\ntraining speed ~= min(loader, gpu). Pick the GPU variant, then give the loader enough workers.")


if __name__ == "__main__":
    main()
