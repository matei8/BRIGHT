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
