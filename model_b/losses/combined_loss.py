import torch
import torch.nn as nn
import torch.nn.functional as F

def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension.
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()

    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted.float()).cumsum(0)

    jaccard = 1.0 - intersection / union

    if p > 1:
        jaccard = torch.cat([
            jaccard[:1],
            jaccard[1:] - jaccard[:-1]
        ])

    return jaccard


def lovasz_softmax_flat(probas, labels, classes="present"):
    """
    Multi-class Lovasz-Softmax loss.
    probas: [P, C]
    labels: [P]
    """
    C = probas.shape[1]
    losses = []

    class_to_sum = (
        range(C) if classes in ["all", "present"] else classes
    )

    for c in class_to_sum:
        fg = (labels == c).float()

        if classes == "present" and fg.sum() == 0:
            continue

        if fg.sum() == 0:
            continue

        errors = (fg - probas[:, c]).abs()

        errors_sorted, perm = torch.sort(
            errors, descending=True
        )

        fg_sorted = fg[perm]

        grad = lovasz_grad(fg_sorted)

        losses.append(torch.dot(errors_sorted, grad))

    if len(losses) == 0:
        return probas.sum() * 0.0

    return torch.stack(losses).mean()


def lovasz_softmax(
    logits,
    labels,
    ignore_index=255
):
    """
    logits: [B, C, H, W]
    labels: [B, H, W]
    """
    probs = F.softmax(logits, dim=1)

    losses = []

    for prob, label in zip(probs, labels):

        valid = label != ignore_index

        prob = prob.permute(1, 2, 0).reshape(-1, prob.shape[0])
        label = label.reshape(-1)

        prob = prob[valid.reshape(-1)]
        label = label[valid.reshape(-1)]

        if label.numel() == 0:
            continue

        losses.append(
            lovasz_softmax_flat(prob, label, classes="present")
        )

    if len(losses) == 0:
        return logits.sum() * 0.0

    return torch.stack(losses).mean()


class CombinedLoss(nn.Module):
    def __init__(
        self,
        lovasz_weight=0.75,
        ignore_index=255
    ):
        super().__init__()

        self.lovasz_weight = lovasz_weight
        self.ce = nn.CrossEntropyLoss(
            ignore_index=ignore_index
        )

    def forward(self, logits, target):

        ce_loss = self.ce(logits, target)

        lovasz_loss = lovasz_softmax(
            logits,
            target,
            ignore_index=255
        )

        total_loss = (
            ce_loss +
            self.lovasz_weight * lovasz_loss
        )

        return total_loss, ce_loss, lovasz_loss


print("Combined CE + Lovasz-Softmax loss defined successfully!")
