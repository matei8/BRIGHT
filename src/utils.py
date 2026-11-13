"""Small run helpers: seeding, device, run directories, checkpoints."""

from __future__ import annotations

import os
import random
import socket
import subprocess
import sys
from datetime import datetime
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
    tmp = str(path) + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)  # atomic: a job killed mid-save never leaves a corrupt checkpoint


def load_checkpoint(path, model, optimizer=None, map_location="cpu") -> dict:
    state = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(state["model"])
    if optimizer is not None and state.get("optimizer") is not None:
        optimizer.load_state_dict(state["optimizer"])
    return state


# --------------------------- resume + provenance ---------------------------

def rng_state() -> dict:
    """Python / numpy / torch RNG state (store in a checkpoint to resume reproducibly)."""
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def set_rng_state(s: dict) -> None:
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"].cpu())
    if s.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([t.cpu() for t in s["cuda"]])


def git_info(root: Path = CODEBASE_ROOT) -> dict:
    def run(*args):
        try:
            return subprocess.check_output(["git", *args], cwd=root, stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return None
    commit = run("rev-parse", "HEAD")
    if commit is None:  # no git (e.g. inside the container): use what the SLURM script recorded on the host
        dirty = os.environ.get("BRIGHT_GIT_DIRTY")
        return {"commit": os.environ.get("BRIGHT_GIT_COMMIT"), "branch": os.environ.get("BRIGHT_GIT_BRANCH"),
                "dirty": None if dirty is None else dirty == "1"}  # None = unknown, not clean
    status = run("status", "--porcelain", "--untracked-files=no")
    return {"commit": commit, "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": None if status is None else bool(status)}


def machine_info(device: torch.device) -> dict:
    return {
        "host": socket.gethostname(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "argv": sys.argv,
        "time": datetime.now().isoformat(timespec="seconds"),
    }
