"""Shared utilities for loss functions."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F


def get_safe_targets(
    labels: torch.Tensor, ignore_index: int | None
) -> tuple[torch.Tensor, torch.Tensor]:
    if ignore_index is None:
        return labels, torch.ones_like(labels, dtype=torch.bool)
    mask = labels != ignore_index
    safe_labels = torch.where(mask, labels, torch.zeros_like(labels))
    return safe_labels, mask


def reduce_spatial(
    loss: torch.Tensor, reduction: str, valid_mask: torch.Tensor | None = None
) -> torch.Tensor:
    if reduction == "none":
        return loss
    loss_flat = loss.view(loss.shape[0], -1)
    if valid_mask is not None and reduction == "mean":
        mask_flat = valid_mask.view(loss.shape[0], -1).to(loss.dtype)
        denom = mask_flat.sum(dim=1).clamp_min(1.0)
        return loss_flat.sum(dim=1) / denom
    if reduction == "mean":
        return loss_flat.mean(dim=-1)
    return loss_flat.sum(dim=-1)


def spatial_one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    one_hot = F.one_hot(labels, num_classes=num_classes)
    return one_hot.movedim(-1, 1).float()


def _wrap_loss(func: Callable, ignore_index: int, **default_kwargs) -> Callable:
    def wrapper(logits: torch.Tensor, labels: torch.Tensor, **kw: Any) -> torch.Tensor:
        return func(
            logits,
            labels,
            ignore_index=ignore_index,
            reduction="none",
            **default_kwargs,
        )

    return wrapper
