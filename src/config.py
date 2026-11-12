from __future__ import annotations

import copy
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

# Environment variable that, when set, overrides [data].dataset_root. Lets a cluster
# job point at a node-local copy of the data without editing any file.
DATA_ROOT_ENV = "BRIGHT_DATA_ROOT"


def _deep_merge(base: dict, override: dict, where: str = "") -> dict:
    """Recursively merge ``override`` into a copy of ``base`` (override wins).

    Strict: every key in ``override`` must already exist in ``base``, so a typo in an
    env profile (``batchsize`` instead of ``batch_size``) fails loudly instead of being
    silently ignored while the base value is used.
    """
    out = copy.deepcopy(base)
    for k, v in override.items():
        if k not in out:
            raise KeyError(f"unknown config key {where + k!r} (not defined in the base config)")
        if isinstance(v, dict) and isinstance(out[k], dict):
            out[k] = _deep_merge(out[k], v, f"{where}{k}.")
        else:
            out[k] = copy.deepcopy(v)
    return out


def _parse_value(text: str):
    """Parse a CLI value with TOML rules (numbers, bools, lists, quoted strings);
    fall back to a bare string, e.g. a path."""
    try:
        return tomllib.loads(f"v = {text}")["v"]
    except tomllib.TOMLDecodeError:
        return text


def _apply_set(cfg: dict, assignment: str) -> None:
    """Apply one ``dotted.key=value`` override in place, e.g. ``data.crop_size=512``."""
    if "=" not in assignment:
        raise ValueError(f"override must look like key=value, got {assignment!r}")
    key, value = assignment.split("=", 1)
    *parents, leaf = key.strip().split(".")
    node = cfg
    for p in parents:
        node = node.get(p) if isinstance(node, dict) else None
        if not isinstance(node, dict):
            raise KeyError(f"cannot set {key!r}: {p!r} is not a config table")
    if leaf not in node:  # strict: catch typos like train.lrr=3e-4
        raise KeyError(f"unknown config key {key.strip()!r} (not defined in the base config)")
    node[leaf] = _parse_value(value.strip())


def load_config(base: str | Path, env: str | Path | None = None,
                overrides: list[str] | None = None) -> dict:
    """Load the layered config used by every script.

    Precedence (later wins):
      1. ``base``      — the experiment definition, shared by all machines.
      2. ``env``       — machine profile, a name (``local`` -> ``config/env/local.toml``)
                         or a path to a TOML file. Only machine-specific keys belong here.
      3. ``$BRIGHT_DATA_ROOT`` — overrides ``data.dataset_root`` if set.
      4. ``overrides`` — ``key=value`` strings from the command line (``--set``).

    Relative paths in any layer are resolved against the codebase root (the parent of
    ``config/``); absolute paths are kept as-is. The resolved root is stored under
    ``_root`` for downstream path handling.
    """
    base = Path(base).resolve()
    root = base.parents[1]
    with open(base, "rb") as f:
        cfg = tomllib.load(f)

    if env:
        env_path = Path(env)
        if env_path.suffix != ".toml":  # a profile name, not a path
            env_path = base.parent / "env" / f"{env}.toml"
        with open(env_path, "rb") as f:
            cfg = _deep_merge(cfg, tomllib.load(f))

    if os.environ.get(DATA_ROOT_ENV):
        cfg.setdefault("data", {})["dataset_root"] = os.environ[DATA_ROOT_ENV]

    for s in overrides or []:
        _apply_set(cfg, s)

    cfg["_root"] = str(root)
    return cfg


def resolve_path(cfg: dict, p: str) -> str:
    """Resolve a config path against the codebase root (absolute paths unchanged)."""
    return str((Path(cfg["_root"]) / os.path.expanduser(p)).resolve())


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
    def from_dict(cls, raw: dict, split: str) -> "DataConfig":
        """Build from an already-loaded (layered) config dict, see ``load_config``."""
        if split not in raw["splits"]:
            raise KeyError(f"split {split!r} not in splits {sorted(raw['splits'])}")
        if split not in raw["augmentation"]:
            raise KeyError(f"split {split!r} missing from [augmentation]")

        data = raw["data"]
        dataset_root = resolve_path(raw, data["dataset_root"])
        missing = [d for d in ("pre-event", "post-event", "target")
                   if not os.path.isdir(os.path.join(dataset_root, d))]
        if missing:
            raise FileNotFoundError(
                f"dataset_root {dataset_root!r} is missing {missing} — set ${DATA_ROOT_ENV} "
                f"or data.dataset_root to the folder holding pre-event/ post-event/ target/")
        return cls(
            split=split,
            dataset_root=dataset_root,
            split_file=resolve_path(raw, raw["splits"][split]),
            crop_size=int(data["crop_size"]),
            augment=bool(raw["augmentation"][split]),
            ignore_index=int(data.get("ignore_index", 255)),
            optical_mean=list(data["optical_norm"]["mean"]),
            optical_std=list(data["optical_norm"]["std"]),
            sar_mean=float(data["sar_norm"]["mean"]),
            sar_std=float(data["sar_norm"]["std"]),
            suffix=str(data.get("suffix", ".tif")),
        )

    @classmethod
    def from_toml(cls, path: str | Path, split: str, env: str | Path | None = None,
                  overrides: list[str] | None = None) -> "DataConfig":
        """Load + resolve in one step. Backward compatible: ``from_toml(path, split)``
        behaves as before (plus ``$BRIGHT_DATA_ROOT`` if set)."""
        return cls.from_dict(load_config(path, env, overrides), split)
