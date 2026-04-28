"""Margin-based losses (CW, DLR)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from ..debug import assert_in_enum
from .common import get_safe_targets, reduce_spatial


def difference_of_logits(
    logits: Tensor, labels: Tensor, ignore_index: int | None = None
) -> Tensor:
    """Difference between correct-class logit and highest other-class logit."""
    if labels.ndim == 2:
        labels = labels.argmax(dim=1)

    labels, valid_mask = get_safe_targets(labels, ignore_index)

    labels_infhot = torch.zeros_like(logits).scatter_(
        1, labels.unsqueeze(1), float("inf")
    )
    class_logits = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
    other_logits = (logits - labels_infhot).amax(dim=1)
    m = class_logits - other_logits
    if ignore_index is not None:
        m = m * valid_mask.to(m.dtype)
    return m


def margin(
    logits: Tensor,
    y_onehot: Tensor,
    delta: float = 0.0,
    ignore_index: int | None = None,
    targeted: bool = False,
) -> Tensor:
    """Output margin adjusted by *delta* for targeted / untargeted attacks."""
    logits_dist = difference_of_logits(logits, y_onehot, ignore_index=ignore_index)
    return delta - logits_dist if targeted else logits_dist + delta


def cw_loss(
    logits: Tensor,
    labels: Tensor,
    delta: float = 0.0,
    targeted: bool = False,
    ignore_index: int = -1,
    reduction: str = "none",
) -> Tensor:
    """Carlini-Wagner (multi-class hinge) loss."""
    assert_in_enum(["mean", "sum", "none"], reduction=reduction)
    m = margin(
        logits, labels, delta=delta, ignore_index=ignore_index, targeted=targeted
    )
    _, mask = get_safe_targets(labels, ignore_index)
    loss = -F.relu(m) * mask.to(m.dtype)
    return reduce_spatial(loss, reduction)


def dlr_loss(
    logits: Tensor,
    labels: Tensor,
    targeted: bool = False,
    ignore_index: int = -1,
    reduction: str = "none",
    eps: float = 1e-12,
) -> Tensor:
    """Difference-of-Logits Ratio (Croce & Hein, 2020)."""
    k = 4 if targeted else 3
    if logits.shape[1] < k:
        raise ValueError(f"logits.shape[1] must be >= {k}")
    logit_dists = difference_of_logits(
        logits=logits, labels=labels, ignore_index=ignore_index
    )
    top_k_logits = torch.topk(logits, k=k, dim=1).values
    if targeted:
        logit_normalization = (
            top_k_logits[:, 0] - (top_k_logits[:, 2] + top_k_logits[:, 3]) / 2
        )
    else:
        logit_normalization = top_k_logits[:, 0] - top_k_logits[:, 2]

    _, mask = get_safe_targets(labels, ignore_index)
    dlr = logit_dists / (logit_normalization + eps)
    loss = dlr * mask.to(dlr.dtype)
    return reduce_spatial(loss, reduction)
