import random
from pathlib import Path
import numpy as np
import imageio.v2 as imageio
import torch
from torch.utils.data import Dataset
from configs.config import OPTICAL_MEAN, OPTICAL_STD, SAR_MEAN, SAR_STD, IGNORE_INDEX, CROP_SIZE

class BrightDataset(Dataset):

    def __init__(self, names, dataset_root, augment=False):
        self.names = names
        self.root = Path(dataset_root)
        self.augment = augment

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):

        name = self.names[index]

        # Load images
        optical = imageio.imread(
            self.root / "pre-event" /
            f"{name}_pre_disaster.tif"
        )[..., :3].astype(np.float32)

        sar = imageio.imread(
            self.root / "post-event" /
            f"{name}_post_disaster.tif"
        ).astype(np.float32)

        target = imageio.imread(
            self.root / "target" /
            f"{name}_building_damage.tif"
        ).astype(np.int64)

        if sar.ndim == 2:
            sar = sar[..., None]

        # Synchronized augmentations
        if self.augment:

            h, w = target.shape

            # Random crop
            top = random.randint(0, h - CROP_SIZE)
            left = random.randint(0, w - CROP_SIZE)

            optical = optical[
                top:top+CROP_SIZE, left:left+CROP_SIZE
            ]
            sar = sar[
                top:top+CROP_SIZE, left:left+CROP_SIZE
            ]
            target = target[
                top:top+CROP_SIZE, left:left+CROP_SIZE
            ]

            # Horizontal flip
            if random.random() < 0.5:
                optical = np.fliplr(optical)
                sar = np.fliplr(sar)
                target = np.fliplr(target)

            # Vertical flip
            if random.random() < 0.5:
                optical = np.flipud(optical)
                sar = np.flipud(sar)
                target = np.flipud(target)

            # Random rotation: 90, 180 or 270 degrees
            k = random.randint(1, 3)
            optical = np.rot90(optical, k)
            sar = np.rot90(sar, k)
            target = np.rot90(target, k)

        # Normalize
        optical = (optical - OPTICAL_MEAN) / OPTICAL_STD
        sar = (sar - SAR_MEAN) / SAR_STD

        # HWC -> CHW
        optical = torch.from_numpy(
            np.ascontiguousarray(optical.transpose(2, 0, 1))
        ).float()

        sar = torch.from_numpy(
            np.ascontiguousarray(sar.transpose(2, 0, 1))
        ).float()

        target = torch.from_numpy(
            np.ascontiguousarray(target)
        ).long()

        return optical, sar, target, name
