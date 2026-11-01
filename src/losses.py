"""Losses for the BRIGHT baseline: CE + 0.75 * Lovasz-Softmax.

The Lovasz-Softmax part is ported from the official repo's
``bda_benchmark/util_func/lovasz_loss.py`` (Maxim Berman 2018, ESAT-PSI KU Leuven,
MIT License) -- multiclass path only, which is all the U-Net uses. Changes are
numerically neutral:
  * ``classes is 'present'`` -> ``classes == 'present'`` (identity check on a str
    literal; SyntaxWarning on Python 3.12+).
  * the deprecated ``torch.autograd.Variable`` wrappers are dropped (no-ops since
    PyTorch 0.4).
  * ``flatten_probas`` indexes with the boolean mask directly instead of
    ``valid.nonzero().squeeze()``, which breaks when exactly one pixel is valid.

``segmentation_loss`` is the repo's U-Net objective, with ``ignore_index`` added to
the CE term (the repo omitted it; labels contain 255 padding, which would otherwise
be an out-of-range target).
"""

try:
    from itertools import ifilterfalse
except ImportError:  # py3k
    from itertools import filterfalse as ifilterfalse

import torch
import torch.nn.functional as F


# --------------------------- LOVASZ-SOFTMAX ---------------------------

def lovasz_grad(gt_sorted):
    """Gradient of the Lovasz extension w.r.t. sorted errors (Alg. 1 in the paper)."""
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_softmax(probas, labels, classes='present', per_image=False, ignore=None):
    """Multi-class Lovasz-Softmax loss.

    probas: [B, C, H, W] class probabilities (after softmax).
    labels: [B, H, W] ground-truth labels in [0, C-1] (``ignore`` excluded).
    classes: 'all', 'present' (classes present in labels), or a list of classes.
    """
    if per_image:
        loss = mean(lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore),
                                        classes=classes)
                    for prob, lab in zip(probas, labels))
    else:
        loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore), classes=classes)
    return loss


def lovasz_softmax_flat(probas, labels, classes='present'):
    """probas: [P, C], labels: [P]."""
    if probas.numel() == 0:
        # only void pixels, the gradients should be 0
        return probas.sum() * 0.
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float()  # foreground for class c
        if classes == 'present' and fg.sum() == 0:
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = (fg - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, lovasz_grad(fg_sorted)))
    return mean(losses)


def flatten_probas(probas, labels, ignore=None):
    """Flatten [B, C, H, W] -> [P, C] and drop ``ignore`` pixels."""
    if probas.dim() == 3:
        # assumes output of a sigmoid layer
        B, H, W = probas.size()
        probas = probas.view(B, 1, H, W)
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B*H*W, C = P, C
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = labels != ignore
    return probas[valid], labels[valid]


# --------------------------- HELPERS ---------------------------

def isnan(x):
    return x != x


def mean(l, ignore_nan=False, empty=0):
    """nanmean compatible with generators."""
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n


# --------------------------- BASELINE OBJECTIVE ---------------------------

def segmentation_loss(logits, labels, ignore_index=255, lovasz_weight=0.75):
    """CE + lovasz_weight * Lovasz-Softmax, both ignoring ``ignore_index``.

    Computed in float32 even under AMP (sorting / cumsum in Lovasz are not
    fp16-friendly).
    """
    logits = logits.float()
    ce = F.cross_entropy(logits, labels, ignore_index=ignore_index)
    lov = lovasz_softmax(F.softmax(logits, dim=1), labels, ignore=ignore_index)
    return ce + lovasz_weight * lov
