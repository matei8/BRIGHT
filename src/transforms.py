"""Image transforms for the BRIGHT 4-channel baseline.

Ported from the official repo's ``bda_benchmark/dataset/imutils.py`` with the
3-channel assumptions removed so that optical (3 ch) and SAR (1 ch) can be
normalized and cropped independently. Flip/rotation are channel-agnostic and are
kept faithful to the repo. Normalization uses ImageNet statistics for the optical
stream (matching the published baseline) and a separate standardization for SAR.
"""

import random

import numpy as np

# Optical normalization — ImageNet stats, as used by the published BRIGHT baseline.
IMAGENET_MEAN = [123.675, 116.28, 103.53]
IMAGENET_STD = [58.395, 57.12, 57.375]

# SAR normalization — PROVISIONAL, estimated over 150 random train tiles (raw 8-bit
# amplitude). Replace with the full-train-split values computed by preprocess/EDA.
SAR_MEAN = 59.121
SAR_STD = 45.478


def normalize_optical(img, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """Standardize a 3-channel optical image (H, W, 3) per channel."""
    img = np.asarray(img, np.float32)
    out = np.empty_like(img, np.float32)

    for i in range(3):
        out[..., i] = (img[..., i] - mean[i]) / std[i]

    return out


def normalize_sar(img, mean=SAR_MEAN, std=SAR_STD):
    """Standardize a single-channel SAR image (H, W, 1)."""
    img = np.asarray(img, np.float32)
    return (img - mean) / std


def random_fliplr(pre_img, post_img, label):
    if random.random() > 0.5:
        pre_img = np.fliplr(pre_img)
        post_img = np.fliplr(post_img)
        label = np.fliplr(label)

    return pre_img, post_img, label


def random_flipud(pre_img, post_img, label):
    if random.random() > 0.5:
        pre_img = np.flipud(pre_img)
        post_img = np.flipud(post_img)
        label = np.flipud(label)

    return pre_img, post_img, label


def random_rot(pre_img, post_img, label):
    k = random.randrange(3) + 1  # 90, 180 or 270 degrees
    pre_img = np.rot90(pre_img, k)
    post_img = np.rot90(post_img, k)
    label = np.rot90(label, k)
    return pre_img, post_img, label


def random_crop(pre_img, post_img, label, crop_size, pad_value=0.0,
                ignore_index=255, cat_max_ratio=0.75):
    """Random ``crop_size`` crop shared across pre/post/label.

    Channel-aware generalization of the repo's ``random_crop``: the padding
    buffers are allocated with each image's own channel count instead of a
    hardcoded 3. Up to 10 candidate windows are tried, preferring a crop that is
    not dominated (>= ``cat_max_ratio``) by a single non-ignore class, which
    oversamples building-bearing regions.
    """
    h, w = label.shape
    H = max(crop_size, h)
    W = max(crop_size, w)
    cp = pre_img.shape[2]
    cq = post_img.shape[2]

    pad_pre = np.full((H, W, cp), pad_value, dtype=np.float32)
    pad_post = np.full((H, W, cq), pad_value, dtype=np.float32)
    pad_label = np.full((H, W), ignore_index, dtype=np.float32)

    h_off = int(np.random.randint(H - h + 1))
    w_off = int(np.random.randint(W - w + 1))

    pad_pre[h_off:h_off + h, w_off:w_off + w, :] = pre_img
    pad_post[h_off:h_off + h, w_off:w_off + w, :] = post_img
    pad_label[h_off:h_off + h, w_off:w_off + w] = label

    h_start = w_start = 0
    for _ in range(10):
        h_start = random.randrange(0, H - crop_size + 1)
        w_start = random.randrange(0, W - crop_size + 1)

        patch = pad_label[h_start:h_start + crop_size, w_start:w_start + crop_size]
        idx, cnt = np.unique(patch, return_counts=True)
        cnt = cnt[idx != ignore_index]

        if cnt.size > 0 and np.max(cnt) / np.sum(cnt) < cat_max_ratio:
            break

    pre_c = pad_pre[h_start:h_start + crop_size, w_start:w_start + crop_size, :]
    post_c = pad_post[h_start:h_start + crop_size, w_start:w_start + crop_size, :]
    label_c = pad_label[h_start:h_start + crop_size, w_start:w_start + crop_size]

    return pre_c, post_c, label_c
