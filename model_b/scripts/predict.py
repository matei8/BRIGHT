
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import imageio.v2 as imageio

from configs.config import (
    DATASET_ROOT,
    OUTPUT_ROOT,
    NUM_CLASSES,
    OPTICAL_MEAN,
    OPTICAL_STD,
    SAR_MEAN,
    SAR_STD,
)
from models.two_stream_fusion import TwoStreamFusionNet
from utils.inference import predict_tiled


CLASS_COLORS = [
    "#000000",  # Background
    "#00AA00",  # Intact
    "#FFFF00",  # Damaged
    "#FF0000",  # Destroyed
]


def colorize_mask(mask):
    cmap = ListedColormap(CLASS_COLORS)
    return cmap(mask)


def load_image(name):
    root = Path(DATASET_ROOT)

    optical = imageio.imread(
        root / "pre-event" / f"{name}_pre_disaster.tif"
    )[..., :3].astype(np.float32)

    sar = imageio.imread(
        root / "post-event" / f"{name}_post_disaster.tif"
    ).astype(np.float32)

    if sar.ndim == 2:
        sar = sar[..., None]

    optical = (optical - np.array(OPTICAL_MEAN)) / np.array(OPTICAL_STD)
    sar = (sar - SAR_MEAN) / SAR_STD

    optical = torch.from_numpy(
        np.ascontiguousarray(optical.transpose(2, 0, 1))
    ).float().unsqueeze(0)

    sar = torch.from_numpy(
        np.ascontiguousarray(sar.transpose(2, 0, 1))
    ).float().unsqueeze(0)

    return optical, sar


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = TwoStreamFusionNet(
        num_classes=NUM_CLASSES
    ).to(device)

    checkpoint = torch.load(
        Path(OUTPUT_ROOT) / "best_model.pth",
        map_location=device
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Replace with a real image name from the BRIGHT dataset.
    image_name = "beirut-explosion_00000030"

    optical, sar = load_image(image_name)

    optical = optical.to(device)
    sar = sar.to(device)

    logits = predict_tiled(
        model,
        optical,
        sar,
        tile_size=512
    )

    prediction = logits.argmax(dim=1)[0].cpu().numpy()

    output_dir = Path(OUTPUT_ROOT) / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 10))
    plt.imshow(colorize_mask(prediction))
    plt.title("Model B - Predicted segmentation")
    plt.axis("off")

    plt.savefig(
        output_dir / f"{image_name}_prediction.png",
        bbox_inches="tight",
        dpi=200
    )
    plt.close()

    print("Prediction saved to:", output_dir)


if __name__ == "__main__":
    main()
