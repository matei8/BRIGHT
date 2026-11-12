"""Recompute SAR normalization stats over a split (default: train only, no leakage).

    python scripts/sar_stats.py [--env local] [--split train]

Prints mean/std of raw post-event SAR (uint8, zeros included) to paste into
[data.sar_norm] of config/baseline.toml. Current values: 56.823 / 45.855.
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402

from src.config import DataConfig, load_config  # noqa: E402


def _tile(path):
    a = np.asarray(imageio.imread(path))
    a = (a[..., 0] if a.ndim == 3 else a).astype(np.float64)
    return a.size, a.sum(), (a * a).sum()


def main():
    p = argparse.ArgumentParser(description="SAR normalization statistics")
    p.add_argument("--config", default=str(ROOT / "config" / "baseline.toml"))
    p.add_argument("--env")
    p.add_argument("--split", default="train")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    dc = DataConfig.from_dict(load_config(args.config, args.env), args.split)
    names = [line.strip() for line in open(dc.split_file) if line.strip()]
    paths = [os.path.join(dc.dataset_root, "post-event", n + "_post_disaster" + dc.suffix) for n in names]
    with ProcessPoolExecutor(args.workers) as ex:
        res = list(ex.map(_tile, paths, chunksize=16))
    n = sum(r[0] for r in res)
    mean = sum(r[1] for r in res) / n
    std = np.sqrt(sum(r[2] for r in res) / n - mean ** 2)
    print(f"{args.split}: {len(paths)} tiles  SAR mean = {mean:.3f}  std = {std:.3f}")


if __name__ == "__main__":
    main()
