"""Training entry point template.

Configure DATASET_ROOT, SPLITS_DIR and OUTPUT_ROOT in configs/config.py.
Run from the project root after installing requirements.
"""
import json, random, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from configs.config import *
from datasets.bright_dataset import BrightDataset
from models.two_stream_fusion import TwoStreamFusionNet
from losses.combined_loss import CombinedLoss
from utils.inference import evaluate_model_tiled

def train_one_epoch(model, loader, criterion, optimizer, device):

    model.train()

    total_loss = 0.0
    total_ce = 0.0
    total_lovasz = 0.0

    progress = tqdm(loader, desc="Training", leave=False)

    for optical, sar, target, names in progress:

        optical = optical.to(device)
        sar = sar.to(device)
        target = target.to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(optical, sar)

        loss, ce_loss, lovasz_loss = criterion(logits, target)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_ce += ce_loss.item()
        total_lovasz += lovasz_loss.item()

        progress.set_postfix(loss=f"{loss.item():.4f}")

    n = len(loader)

    return {
        "loss": total_loss / n,
        "ce_loss": total_ce / n,
        "lovasz_loss": total_lovasz / n
    }



def fit_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    epochs,
    output_dir
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_miou = -1.0
    history = []

    for epoch in range(1, epochs + 1):

        start_time = time.time()

        # Training
        train_results = train_one_epoch(
            model, train_loader,
            criterion, optimizer, device
        )

        # Validation
        val_results = evaluate_model_tiled(
            model, val_loader,
            criterion, device, tile_size=512
        )

        epoch_time = time.time() - start_time

        record = {
            "epoch": epoch,
            "train_loss": train_results["loss"],
            "val_loss": val_results["loss"],
            "val_miou": val_results["miou"],
            "val_oa": val_results["oa"],
            "val_iou_per_class": val_results["iou_per_class"].tolist(),
            "val_f1_per_class": val_results["f1_per_class"].tolist(),
            "epoch_time_seconds": epoch_time
        }

        history.append(record)

        print(f"\nEpoch {epoch}/{epochs}")
        print(f"Train Loss: {train_results['loss']:.4f}")
        print(f"Val Loss:   {val_results['loss']:.4f}")
        print(f"Val mIoU:   {val_results['miou']:.4f}")
        print(f"Val OA:     {val_results['oa']:.4f}")
        print(f"Time:       {epoch_time:.1f}s")

        # Save latest checkpoint
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_miou": val_results["miou"]
        }, output_dir / "last_model.pth")

        # Save best checkpoint
        if val_results["miou"] > best_miou:
            best_miou = val_results["miou"]

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_miou": best_miou
            }, output_dir / "best_model.pth")

            print("New best model saved!")

        # Save training history
        with open(output_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=4)

    print("\nTraining completed!")
    print("Best validation mIoU:", best_miou)

    return history




def load_split(path):
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]


def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits = {k: load_split(SPLITS_DIR / f"{k}_set.txt") for k in ("train", "val")}
    train_ds = BrightDataset(splits["train"], DATASET_ROOT, augment=True)
    val_ds = BrightDataset(splits["val"], DATASET_ROOT, augment=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
    model = TwoStreamFusionNet(num_classes=NUM_CLASSES).to(device)
    criterion = CombinedLoss(lovasz_weight=LOVASZ_WEIGHT, ignore_index=IGNORE_INDEX).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    fit_model(model, train_loader, val_loader, criterion, optimizer, device, EPOCHS, OUTPUT_ROOT)


if __name__ == "__main__":
    main()
