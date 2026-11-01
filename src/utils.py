"""Small run helpers: seeding, device, run directories, checkpoints."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

CODEBASE_ROOT = Path(__file__).resolve().parents[1]


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed python / numpy / torch (CPU + CUDA). ``deterministic`` trades speed
    for run-to-run reproducibility of cuDNN kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def seed_worker(worker_id: int) -> None:
    """DataLoader ``worker_init_fn``: torch seeds each worker (and python's
    ``random``), but not numpy -- which ``random_crop`` uses. Derive it here."""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_run_dirs(run_name: str, root: Path = CODEBASE_ROOT) -> tuple[Path, Path]:
    """Create and return ``results/<run>/`` and ``checkpoints/<run>/``."""
    results_dir = root / "results" / run_name
    ckpt_dir = root / "checkpoints" / run_name
    results_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return results_dir, ckpt_dir


def save_checkpoint(path, model, optimizer=None, epoch=None, best_metric=None, extra=None) -> None:
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "best_metric": best_metric,
        "extra": extra or {},
    }
    torch.save(state, path)


def load_checkpoint(path, model, optimizer=None, map_location="cpu") -> dict:
    state = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(state["model"])
    if optimizer is not None and state.get("optimizer") is not None:
        optimizer.load_state_dict(state["optimizer"])
    return state
