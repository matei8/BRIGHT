"""BRIGHT dataset — 4-channel early-fusion (optical RGB + single SAR band).

Mirrors the official repo's ``MultimodalDamageAssessmentDatset`` but keeps SAR as
a single channel (the repo replicated it to 3 to make a 6-channel input). Each
item returns the two streams separately; the model input is formed downstream as
``torch.cat([optical, sar], dim=1)`` -> 4 channels.

All settings come from a single ``DataConfig`` (see src/config.py), loaded from
``config/data_config.toml``.
"""

import os

import imageio.v2 as imageio
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from . import transforms as T
from .config import DataConfig


class BrightDataset(Dataset):
    def __init__(self, config: DataConfig):
        self.cfg = config
        with open(config.split_file) as f:
            self.names = [line.strip() for line in f if line.strip()]

    def __len__(self):
        return len(self.names)

    def _load(self, name):
        cfg = self.cfg

        pre_p = os.path.join(cfg.dataset_root, 'pre-event', name + '_pre_disaster' + cfg.suffix)
        post_p = os.path.join(cfg.dataset_root, 'post-event', name + '_post_disaster' + cfg.suffix)
        tgt_p = os.path.join(cfg.dataset_root, 'target', name + '_building_damage' + cfg.suffix)

        pre = np.asarray(imageio.imread(pre_p), np.float32)[:, :, 0:3]  # optical RGB (H,W,3)
        post = np.asarray(imageio.imread(post_p), np.float32)           # SAR (H,W)

        if post.ndim == 2:
            post = post[:, :, None]                                     # -> (H,W,1), NOT replicated

        tgt = np.asarray(imageio.imread(tgt_p))                         # labels (H,W)
        return pre, post, tgt

    def __getitem__(self, index):
        cfg = self.cfg
        name = self.names[index]
        pre, post, clf = self._load(name)

        if cfg.augment:
            pre, post, clf = T.random_crop(pre, post, clf, cfg.crop_size, ignore_index=cfg.ignore_index)
            pre, post, clf = T.random_fliplr(pre, post, clf)
            pre, post, clf = T.random_flipud(pre, post, clf)
            pre, post, clf = T.random_rot(pre, post, clf)

        pre = T.normalize_optical(pre, cfg.optical_mean, cfg.optical_std)
        post = T.normalize_sar(post, cfg.sar_mean, cfg.sar_std)

        # to contiguous CHW float32 tensors
        optical = torch.from_numpy(np.ascontiguousarray(np.transpose(pre, (2, 0, 1)), np.float32))
        sar = torch.from_numpy(np.ascontiguousarray(np.transpose(post, (2, 0, 1)), np.float32))

        clf = np.ascontiguousarray(clf).astype(np.int64)
        loc = clf.copy()
        loc[(loc == 2) | (loc == 3)] = 1  # damaged/destroyed -> building; ignore(255) untouched

        return optical, sar, torch.from_numpy(loc), torch.from_numpy(clf), name


def build_loader(dataset, batch_size, shuffle, num_workers=0, drop_last=False, pin_memory=True):
    """Thin DataLoader wrapper. num_workers defaults to 0 for Windows safety."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      drop_last=drop_last, pin_memory=pin_memory)
