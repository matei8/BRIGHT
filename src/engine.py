"""Shared train / evaluate loops, reused unchanged by variants A-E.

Variant-specific wiring (how optical + SAR reach the model) is isolated in a
``forward_fn(model, optical, sar) -> logits`` callable; variant A uses
``early_fusion_forward`` (channel concat -> 4-ch U-Net).

``evaluate`` mirrors the official repo's ``validation()`` / ``test()`` exactly
(three-evaluator pattern), except that each full tile is predicted in
``tile``-sized windows forwarded one at a time (8 GB VRAM budget) and stitched
back before scoring -- every labelled pixel is still scored. ``tile=0`` forwards the
full tile at once, exactly like the repo (use it when VRAM allows, e.g. on the cluster).

It also reports event-level mIoU (mean over disaster events of per-event mIoU, as in
the official infer_UNet.py) and the per-event / per-disaster-type breakdown.
"""

from __future__ import annotations

import time

import numpy as np
import torch
from tqdm import tqdm

from .metrics import DISASTER_TYPES, EVENTS, Evaluator, GroupEvaluator


def to_model_format(model, x):
    """Match the input memory layout to the model's (see ``use_channels_last``)."""
    if getattr(model, "channels_last", False):
        return x.contiguous(memory_format=torch.channels_last)
    return x


def use_channels_last(model):
    """Switch a model to NHWC (channels_last) memory format. Same math, but lets tensor
    cores run the convolutions: ~1.9x faster training on H100 in bf16 (measured)."""
    model.to(memory_format=torch.channels_last)
    model.channels_last = True
    return model


def early_fusion_forward(model, optical, sar):
    """Variant A: concat optical (B,3,H,W) + SAR (B,1,H,W) -> (B,4,H,W)."""
    return model(to_model_format(model, torch.cat([optical, sar], dim=1)))


AMP_DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}


def train_one_epoch(model, loader, optimizer, device, loss_fn, forward_fn=early_fusion_forward,
                    scaler=None, max_steps=None, ignore_index=255, desc="train", amp_dtype=None,
                    should_stop=None, disable_pbar=False, timing=None):
    """One pass over ``loader``. Mixed precision: pass ``amp_dtype`` (fp16 needs a
    ``scaler``, bf16 does not). ``should_stop()`` is polled every step (SLURM SIGTERM);
    when it returns True the epoch is abandoned and ``None`` is returned. If a
    ``timing`` dict is given, it receives 'steps', 'seconds' and 'data_wait' (time spent
    waiting for the DataLoader -- high means the input pipeline is the bottleneck)."""
    model.train()
    if amp_dtype is None and scaler is not None:  # backward compatible: scaler => fp16
        amp_dtype = torch.float16
    use_amp = amp_dtype is not None and device.type == "cuda"
    total, n = 0.0, 0

    pbar = tqdm(loader, desc=desc, leave=False, total=min(max_steps, len(loader)) if max_steps else len(loader),
                disable=disable_pbar)
    t_start = t_fetch = time.time()
    data_wait = 0.0
    for step, (optical, sar, _loc, clf, _names) in enumerate(pbar):
        data_wait += time.time() - t_fetch
        if max_steps and step >= max_steps:
            break
        if should_stop is not None and should_stop():
            return None

        clf = clf.to(device, non_blocking=True).long()
        if not (clf != ignore_index).any():  # all-ignore batch (mirrors repo)
            continue
        optical = optical.to(device, non_blocking=True)
        sar = sar.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            logits = forward_fn(model, optical, sar)
        loss = loss_fn(logits, clf)  # computed in fp32 inside the loss

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total += loss.item()
        n += 1
        pbar.set_postfix(loss=f"{total / n:.4f}")
        t_fetch = time.time()

    if timing is not None:
        timing.update(steps=n, seconds=time.time() - t_start, data_wait=data_wait)

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
             desc="eval", limit=0, on_batch=None, disable_pbar=False):
    """Repo-identical metric suite over a full split.

    Returns dict with loc_f1, clf_f1 (harmonic mean of damage F1 over classes
    1..3), damage_f1_per_class, oa, miou (nanmean IoU over all 4 classes incl.
    background -- the paper's set-level mIoU), iou_per_class, event_miou (mean over
    events present), per_event, per_type. Values in [0, 1]. ``limit`` > 0 stops after
    that many tiles (quick in-loop checks); ``on_batch(pred, names)`` sees predictions.
    """
    was_training = model.training
    model.eval()
    ev_loc, ev_clf, ev_total = Evaluator(2), Evaluator(num_classes), Evaluator(num_classes)
    ev_event, ev_type = GroupEvaluator(EVENTS, num_classes), GroupEvaluator(DISASTER_TYPES, num_classes)
    seen = 0

    for optical, sar, loc, clf, names in tqdm(loader, desc=desc, leave=False, disable=disable_pbar):
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
        ev_event.add(labels_clf, out_clf, names)
        ev_type.add(labels_clf, out_clf, names)
        if on_batch is not None:
            on_batch(out_clf, names)
        seen += len(names)
        if limit and seen >= limit:
            break

    model.train(was_training)

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
        "event_miou": ev_event.mean_miou(),
        "per_event": ev_event.per_group(),
        "per_type": ev_type.per_group(),
        "tiles": seen,
    }
