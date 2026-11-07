import numpy as np

import numpy as np
import torch

NUM_CLASSES = 4
CLASS_NAMES = ["Background", "Intact", "Damaged", "Destroyed"]


class SegmentationMetrics:
    def __init__(self, num_classes=4, ignore_index=255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion_matrix = np.zeros(
            (num_classes, num_classes), dtype=np.int64
        )

    def update(self, logits, target):
        predictions = logits.argmax(dim=1)

        predictions = predictions.detach().cpu().numpy().reshape(-1)
        target = target.detach().cpu().numpy().reshape(-1)

        valid = (
            (target != self.ignore_index) &
            (target >= 0) &
            (target < self.num_classes)
        )

        target = target[valid]
        predictions = predictions[valid]

        indices = target * self.num_classes + predictions

        self.confusion_matrix += np.bincount(
            indices,
            minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)

    def compute(self):
        cm = self.confusion_matrix

        true_positive = np.diag(cm)
        actual = cm.sum(axis=1)
        predicted = cm.sum(axis=0)

        union = actual + predicted - true_positive

        iou = np.divide(
            true_positive,
            union,
            out=np.full(self.num_classes, np.nan),
            where=union != 0
        )

        f1 = np.divide(
            2 * true_positive,
            actual + predicted,
            out=np.full(self.num_classes, np.nan),
            where=(actual + predicted) != 0
        )

        total = cm.sum()
        oa = true_positive.sum() / total if total > 0 else 0.0

        return {
            "iou_per_class": iou,
            "f1_per_class": f1,
            "miou": float(np.nanmean(iou)),
            "oa": float(oa),
            "confusion_matrix": cm.copy()
        }

