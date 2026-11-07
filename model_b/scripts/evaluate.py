
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from configs.config import (
    DATASET_ROOT,
    SPLITS_DIR,
    OUTPUT_ROOT,
    NUM_CLASSES,
    IGNORE_INDEX,
)
from datasets.bright_dataset import BrightDataset
from models.two_stream_fusion import TwoStreamFusionNet
from losses.combined_loss import CombinedLoss
from utils.inference import evaluate_model_tiled


def load_split(path):
    return [
        line.strip()
        for line in Path(path).read_text().splitlines()
        if line.strip()
    ]


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    test_names = load_split(SPLITS_DIR / "test_set.txt")

    test_dataset = BrightDataset(
        test_names,
        DATASET_ROOT,
        augment=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    model = TwoStreamFusionNet(
        num_classes=NUM_CLASSES
    ).to(device)

    checkpoint_path = OUTPUT_ROOT / "best_model.pth"

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    criterion = CombinedLoss(
        lovasz_weight=0.75,
        ignore_index=IGNORE_INDEX
    ).to(device)

    results = evaluate_model_tiled(
        model,
        test_loader,
        criterion,
        device,
        tile_size=512
    )

    output_dir = Path(OUTPUT_ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)

    serializable_results = {
        "test_loss": float(results["loss"]),
        "test_ce_loss": float(results["ce_loss"]),
        "test_lovasz_loss": float(results["lovasz_loss"]),
        "test_miou": float(results["miou"]),
        "test_oa": float(results["oa"]),
        "iou_per_class": results["iou_per_class"].tolist(),
        "f1_per_class": results["f1_per_class"].tolist(),
        "confusion_matrix": results["confusion_matrix"].tolist()
    }

    with open(output_dir / "test_metrics.json", "w") as f:
        json.dump(serializable_results, f, indent=4)

    print("\nTest results:")
    print("mIoU:", results["miou"])
    print("OA:", results["oa"])
    print("IoU per class:", results["iou_per_class"])
    print("F1 per class:", results["f1_per_class"])


if __name__ == "__main__":
    main()
