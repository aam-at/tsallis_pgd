from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from torch import Tensor


class ConfusionMatrix:
    """A class to compute and represent the confusion matrix for classification
    tasks."""

    def __init__(self, num_classes: int, ignore_index: Optional[int] = None):
        """Initialize the confusion matrix.

        :param num_classes: Number of classes in the classification
            task.
        :param ignore_index: Index to ignore.
        """
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.mat = None

    def update(self, true_labels: Tensor, pred_labels: Tensor):
        """Update the confusion matrix with new batch of labels.

        :param true_labels: True labels.
        :param pred_labels: Predicted labels.
        """
        if self.mat is None:
            self.mat = torch.zeros(
                (self.num_classes, self.num_classes),
                dtype=torch.int64,
                device=true_labels.device,
            )

        with torch.no_grad():
            valid = (true_labels >= 0) & (true_labels < self.num_classes)
            if self.ignore_index is not None:
                valid &= true_labels != self.ignore_index
            indices = (
                self.num_classes * true_labels[valid].to(torch.int64)
                + pred_labels[valid]
            )
            self.mat += torch.bincount(indices, minlength=self.num_classes**2).reshape(
                self.num_classes, self.num_classes
            )

    def reset(self):
        """Reset the confusion matrix to zeros."""
        if self.mat is not None:
            self.mat.zero_()

    def compute(self):
        """Compute global accuracy, per-class accuracy, and Intersection over
        Union (IoU).

        :return: Tuple of (global_accuracy, per_class_accuracy, IoU).
        """
        assert self.mat is not None, "Confusion matrix is not initialized."
        h = self.mat
        acc_global = torch.diag(h).sum() / h.sum()
        acc = torch.diag(h) / h.sum(1)
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        return acc_global, acc, iu

    def __str__(self):
        acc_global, accs, ious = self.compute()
        return (
            "Per-class Acc: " + "|".join(f"{acc:>5.2%}" for acc in accs.tolist()) + "\n"
            "IoUs         : " + "|".join(f"{iou:>5.2%}" for iou in ious.tolist()) + "\n"
            f"Global Acc   : {acc_global.item():.2%}\n"
            f"Mean IoU     : {ious.nanmean().item():.2%}"
        )


def color_map(N=256, normalized=False) -> np.ndarray:
    """Generate a color map.

    :param N: Number of colors in the map.
    :param normalized: Whether to normalize the colors to [0, 1].
    :return: A color map as an N x 3 array.
    """
    cmap = np.zeros((N, 3), dtype=np.float32 if normalized else np.uint8)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            shift = 7 - j
            r |= ((c >> 0) & 1) << shift
            g |= ((c >> 1) & 1) << shift
            b |= ((c >> 2) & 1) << shift
            c >>= 3
        cmap[i] = (r, g, b)
    return cmap / 255.0 if normalized else cmap


def pred_to_image(prediction: Tensor, cmap: Optional[np.ndarray] = None) -> Tensor:
    """Convert prediction tensor to an image tensor using a color map.

    :param prediction: Prediction tensor.
    :param cmap: Optional color map. If None, a default color map is
        used.
    :return: Image tensor.
    """
    if cmap is None:
        cmap = torch.tensor(
            color_map(normalized=True), dtype=torch.float, device=prediction.device
        )
    pred_image = cmap[prediction].permute(-1, 0, 1)  # Change channel position
    return pred_image


def create_label_lut(label_ids: List[int], ignore_index: int = 255) -> List[int]:
    """Create a lookup table for label mapping.

    :param label_ids: List of label IDs.
    :param ignore_index: Index to ignore.
    :return: Lookup table.
    """
    label_lut = [ignore_index] * 256
    for i, label in enumerate(label_ids):
        label_lut[label] = i
    return label_lut


def label_map(img: Image.Image, label_lut: List[int]) -> Image.Image:
    """Map labels in an image using a lookup table.

    :param img: Input image.
    :param label_lut: Lookup table for label mapping.
    :return: Mapped image.
    """
    return img.point(label_lut)
