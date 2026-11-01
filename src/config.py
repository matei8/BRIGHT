from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataConfig:
    """All settings a BrightDataset needs, resolved for one split."""

    split: str                 # 'train' | 'val' | 'test'
    dataset_root: str          # absolute path to the folder with pre-event/ post-event/ target/
    split_file: str            # absolute path to the split's name list
    crop_size: int             # random-crop size (used only when augment is True)
    augment: bool              # apply random crop/flip/rotation
    ignore_index: int          # padded / nodata label value
    optical_mean: list[float]
    optical_std: list[float]
    sar_mean: float
    sar_std: float
    suffix: str = ".tif"

    @classmethod
    def from_toml(cls, path: str | Path, split: str) -> "DataConfig":
        path = Path(path)
        with open(path, "rb") as f:
            raw = tomllib.load(f)

        if split not in raw["splits"]:
            raise KeyError(f"split {split!r} not in splits {sorted(raw['splits'])}")
        if split not in raw["augmentation"]:
            raise KeyError(f"split {split!r} missing from [augmentation]")

        # TOML paths are relative to the codebase root (parent of this config/ dir).
        base = path.resolve().parents[1]
        data = raw["data"]
        return cls(
            split=split,
            dataset_root=str((base / data["dataset_root"]).resolve()),
            split_file=str((base / raw["splits"][split]).resolve()),
            crop_size=int(data["crop_size"]),
            augment=bool(raw["augmentation"][split]),
            ignore_index=int(data.get("ignore_index", 255)),
            optical_mean=list(data["optical_norm"]["mean"]),
            optical_std=list(data["optical_norm"]["std"]),
            sar_mean=float(data["sar_norm"]["mean"]),
            sar_std=float(data["sar_norm"]["std"]),
            suffix=str(data.get("suffix", ".tif")),
        )


@dataclass
class TrainConfig:
    """Training / evaluation / runtime settings, from [train] and [runtime]."""

    run_name: str
    seed: int
    epochs: int
    max_steps: int             # 0 = full epochs
    train_batch_size: int
    eval_batch_size: int
    eval_tile_size: int        # sliding-window size for val/test inference
    learning_rate: float
    weight_decay: float
    lovasz_weight: float
    num_classes: int
    in_channels: int
    val_every: int
    amp: bool
    deterministic: bool
    num_workers: int
    pin_memory: bool

    @classmethod
    def from_toml(cls, path: str | Path) -> "TrainConfig":
        with open(path, "rb") as f:
            raw = tomllib.load(f)
        t, r = raw["train"], raw.get("runtime", {})
        return cls(
            run_name=str(t["run_name"]),
            seed=int(t.get("seed", 42)),
            epochs=int(t["epochs"]),
            max_steps=int(t.get("max_steps", 0)),
            train_batch_size=int(t["train_batch_size"]),
            eval_batch_size=int(t.get("eval_batch_size", 1)),
            eval_tile_size=int(t.get("eval_tile_size", 512)),
            learning_rate=float(t["learning_rate"]),
            weight_decay=float(t["weight_decay"]),
            lovasz_weight=float(t.get("lovasz_weight", 0.75)),
            num_classes=int(t.get("num_classes", 4)),
            in_channels=int(t.get("in_channels", 4)),
            val_every=int(t.get("val_every", 1)),
            amp=bool(r.get("amp", True)),
            deterministic=bool(r.get("deterministic", True)),
            num_workers=int(r.get("num_workers", 0)),
            pin_memory=bool(r.get("pin_memory", True)),
        )
