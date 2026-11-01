"""Shared train / evaluate loops, reused unchanged by variants A-E.

Variant-specific wiring (how optical + SAR reach the model) is isolated in a
``forward_fn(model, optical, sar) -> logits`` callable; variant A uses
``early_fusion_forward`` (channel concat -> 4-ch U-Net).

``evaluate`` mirrors the official repo's ``validation()`` / ``test()`` exactly
(three-evaluator pattern), except that each full tile is predicted in
``tile``-sized windows forwarded one at a time (8 GB VRAM budget) and stitched
back before scoring -- every labelled pixel is still scored.
"""

from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm

from .metrics import Evaluator


def early_fusion_forward(model, optical, sar):
    """Variant A: concat optical (B,3,H,W) + SAR (B,1,H,W) -> (B,4,H,W)."""
    return model(torch.cat([optical, sar], dim=1))


def train_one_epoch(model, loader, optimizer, device, loss_fn, forward_fn=early_fusion_forward,
                    scaler=None, max_steps=None, ignore_index=255, desc="train"):
    model.train()
    use_amp = scaler is not None
    total, n = 0.0, 0

    pbar = tqdm(loader, desc=desc, leave=False, total=max_steps or len(loader))
    for step, (optical, sar, _loc, clf, _names) in enumerate(pbar):
        if max_steps and step >= max_steps:
            break

        clf = clf.to(device, non_blocking=True).long()
        if not (clf != ignore_index).any():  # all-ignore batch (mirrors repo)
            continue
        optical = optical.to(device, non_blocking=True)
        sar = sar.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = forward_fn(model, optical, sar)
        loss = loss_fn(logits, clf)  # computed in fp32 inside the loss

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total += loss.item()
        n += 1
        pbar.set_postfix(loss=f"{total / n:.4f}")

    return total / max(n, 1)


@torch.no_grad()
def predict_tiled(model, optical, sar, forward_fn=early_fusion_forward, tile=512):
    """Argmax prediction (B,H,W) for full tiles, forwarding ``tile`` x ``tile``
    windows one at a time. Windows step by ``tile``; if H/W is not a multiple of
    ``tile`` the last window is anchored to the border (overlap is overwritten)."""
    B, _, H, W = optical.shape
    if tile is None or tile <= 0 or (H <= tile and W <= tile):
        return forward_fn(model, optical, sar).argmax(dim=1)

    def starts(size):
        s = list(range(0, max(size - tile, 0) + 1, tile))
        if s[-1] + tile < size:
            s.append(size - tile)
        return s

    pred = torch.empty((B, H, W), dtype=torch.long, device=optical.device)
    for y in starts(H):
        for x in starts(W):
            o = optical[:, :, y:y + tile, x:x + tile]
            s = sar[:, :, y:y + tile, x:x + tile]
            pred[:, y:y + tile, x:x + tile] = forward_fn(model, o, s).argmax(dim=1)
    return pred


@torch.no_grad()
def evaluate(model, loader, device, forward_fn=early_fusion_forward, num_classes=4, tile=512,
             desc="eval"):
    """Repo-identical metric suite over a full split.

    Returns dict with loc_f1, clf_f1 (harmonic mean of damage F1 over classes
    1..3), damage_f1_per_class, oa, miou (nanmean IoU over all 4 classes incl.
    background -- the paper's set-level mIoU), iou_per_class. Values in [0, 1].
    """
    model.eval()
    ev_loc, ev_clf, ev_total = Evaluator(2), Evaluator(num_classes), Evaluator(num_classes)

    for optical, sar, loc, clf, _names in tqdm(loader, desc=desc, leave=False):
        optical = optical.to(device, non_blocking=True)
        sar = sar.to(device, non_blocking=True)

        out_clf = predict_tiled(model, optical, sar, forward_fn, tile).cpu().numpy()
        labels_loc = loc.numpy()
        labels_clf = clf.numpy()

        out_loc = out_clf.copy()
        out_loc[out_loc > 0] = 1
        ev_loc.add_batch(labels_loc, out_loc)

        bmask = labels_loc > 0  # note: 255 (ignore) is also > 0, as in the repo;
        ev_clf.add_batch(labels_clf[bmask], out_clf[bmask])  # Evaluator drops 255 anyway
        ev_total.add_batch(labels_clf, out_clf)

    damage_f1 = ev_clf.Damage_F1_score()
    with np.errstate(divide="ignore"):
        clf_f1 = len(damage_f1) / np.sum(1.0 / damage_f1)  # 0 if any class F1 is 0

    return {
        "loc_f1": float(ev_loc.Pixel_F1_score()),
        "clf_f1": float(clf_f1),
        "damage_f1_per_class": [float(v) for v in damage_f1],
        "oa": float(ev_total.Pixel_Accuracy()),
        "miou": float(ev_total.Mean_Intersection_over_Union()),
        "iou_per_class": [float(v) for v in ev_total.Intersection_over_Union()],
        "confusion_matrix": ev_total.confusion_matrix.tolist(),
    }
