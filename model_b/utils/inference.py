import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from utils.metrics import SegmentationMetrics
from configs.config import NUM_CLASSES, IGNORE_INDEX

@torch.no_grad()
def predict_tiled(model, optical, sar, tile_size=512):
    """
    Predicts a complete image using non-overlapping 512x512 windows.
    optical: [B, 3, H, W]
    sar:     [B, 1, H, W]
    returns: [B, 4, H, W]
    """
    model.eval()

    B, _, H, W = optical.shape

    # Pad dimensions if they are not divisible by tile_size
    pad_h = (tile_size - H % tile_size) % tile_size
    pad_w = (tile_size - W % tile_size) % tile_size

    if pad_h or pad_w:
        optical = F.pad(optical, (0, pad_w, 0, pad_h))
        sar = F.pad(sar, (0, pad_w, 0, pad_h))

    padded_h, padded_w = optical.shape[-2:]

    predictions = torch.zeros(
        (B, NUM_CLASSES, padded_h, padded_w),
        device=optical.device
    )

    for top in range(0, padded_h, tile_size):
        for left in range(0, padded_w, tile_size):

            optical_patch = optical[
                :, :, top:top+tile_size, left:left+tile_size
            ]

            sar_patch = sar[
                :, :, top:top+tile_size, left:left+tile_size
            ]

            patch_logits = model(optical_patch, sar_patch)

            predictions[
                :, :, top:top+tile_size, left:left+tile_size
            ] = patch_logits

    return predictions[:, :, :H, :W]



@torch.no_grad()
def evaluate_model_tiled(model, loader, criterion, device, tile_size=512):

    model.eval()

    metrics = SegmentationMetrics(
        num_classes=NUM_CLASSES,
        ignore_index=IGNORE_INDEX
    )

    total_loss = 0.0
    total_ce = 0.0
    total_lovasz = 0.0

    for optical, sar, target, names in tqdm(
        loader, desc="Tiled validation"
    ):

        optical = optical.to(device, non_blocking=True)
        sar = sar.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        logits = predict_tiled(
            model, optical, sar, tile_size=tile_size
        )

        loss, ce_loss, lovasz_loss = criterion(logits, target)

        total_loss += loss.item()
        total_ce += ce_loss.item()
        total_lovasz += lovasz_loss.item()

        metrics.update(logits, target)

    results = metrics.compute()

    n = len(loader)
    results["loss"] = total_loss / n
    results["ce_loss"] = total_ce / n
    results["lovasz_loss"] = total_lovasz / n

    return results

