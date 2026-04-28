"""KL divergence, Dice loss, and Jensen-Shannon divergence."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from ..debug import assert_in_enum
from .common import get_safe_targets, reduce_spatial, spatial_one_hot


def kl_divergence(
    logits_p: Tensor,
    logits_q: Tensor,
    labels: Tensor | None = None,
    ignore_index: int | None = None,
) -> Tensor:
    """KL divergence between two sets of logits."""
    if logits_p.shape != logits_q.shape:
        raise ValueError(
            f"logits_p and logits_q must have same shape, got {logits_p.shape} and {logits_q.shape}"
        )
    if logits_p.ndim < 2:
        raise ValueError(
            f"logits_p/logits_q must have at least 2 dims (N,C,...) got {logits_p.ndim}"
        )
    class_dim = 1
    p = F.softmax(logits_p, dim=class_dim)
    log_p = F.log_softmax(logits_p, dim=class_dim)
    log_q = F.log_softmax(logits_q, dim=class_dim)
    kl = (p * (log_p - log_q)).sum(dim=class_dim)
    if ignore_index is not None:
        if labels is None:
            raise ValueError("labels must be provided when ignore_index is set")
        if labels.shape != kl.shape:
            raise ValueError(
                "labels shape must match KL output shape when ignore_index is set, "
                f"got labels {labels.shape} and kl {kl.shape}"
            )
        kl = kl.masked_fill(labels == ignore_index, 0.0)
    return kl


def dice_loss(
    logits: Tensor,
    labels: Tensor,
    *,
    ignore_index: Optional[int] = None,
    class_weights: Optional[list[float]] = None,
    squared_pred: bool = False,
    smooth_nr: float = 1e-5,
    smooth_dr: float = 1e-5,
    eps: float = 1e-7,
    reduction: str = "mean",
    dims: Optional[Tuple[int, ...]] = None,
) -> Tensor:
    """Multiclass Dice loss (assumes logits input)."""
    if logits.ndim < 3:
        raise ValueError("logits must be of shape (N, C, ...)")
    if labels.ndim != logits.ndim - 1:
        raise ValueError("labels must be (N, ...) with class indices")
    if reduction not in {"mean", "sum", "none"}:
        raise ValueError("Invalid reduction: choose from {'mean','sum','none'}")

    n, c = logits.shape[:2]
    probs = F.softmax(logits, dim=1)

    safe_target, valid = get_safe_targets(labels, ignore_index)
    target_1h = spatial_one_hot(safe_target, c).to(probs.dtype)
    mask_f = valid.unsqueeze(1).to(probs.dtype)
    probs = probs * mask_f
    target_1h = target_1h * mask_f

    if dims is None:
        dims = tuple(range(2, probs.ndim))

    intersection = torch.sum(probs * target_1h, dim=dims)
    pred_sum = (
        torch.sum(probs * probs, dim=dims)
        if squared_pred
        else torch.sum(probs, dim=dims)
    )
    targ_sum = (
        torch.sum(target_1h * target_1h, dim=dims)
        if squared_pred
        else torch.sum(target_1h, dim=dims)
    )
    dice = (2 * intersection + smooth_nr) / (pred_sum + targ_sum + smooth_dr + eps)
    loss_per_class = 1.0 - dice

    if class_weights is not None:
        cw = torch.as_tensor(
            class_weights, dtype=loss_per_class.dtype, device=loss_per_class.device
        )
        if cw.numel() != c:
            raise ValueError(f"class_weights length {cw.numel()} != num classes {c}")
        loss_per_class = loss_per_class * cw.unsqueeze(0)

    loss_per_sample = loss_per_class.mean(dim=1)

    if reduction == "mean":
        return loss_per_sample
    else:
        expand_shape = (n,) + (1,) * (labels.ndim - 1)
        base_map = loss_per_sample.view(expand_shape).to(probs.dtype) * valid.to(
            probs.dtype
        )
        if reduction == "sum":
            num_valid = valid.view(n, -1).sum(dim=1).clamp_min(1)
            scale = num_valid.view(expand_shape)
            return base_map * scale
        else:
            return base_map


def js_div(
    p: Tensor,
    q: Tensor,
    weights: Tensor | None = None,
    softmax_output: bool = False,
    ignore_index: int = -1,
    red_dim: int | tuple[int, ...] | None = None,
    reduction: str = "none",
) -> Tensor:
    """JS divergence between predictions *p* (logits) and labels *q*."""
    if not softmax_output:
        p = F.softmax(p, 1)
    q_safe, valid_mask = get_safe_targets(q, ignore_index)
    if reduction != "none" and valid_mask.sum() > 0:
        raise ValueError("Incompatible setup.")
    q_onehot = spatial_one_hot(q_safe, p.shape[1]).to(p.dtype)

    m = (p + q_onehot) / 2
    loss = (
        F.kl_div(m.log(), p, reduction=reduction)
        + F.kl_div(m.log(), q_onehot, reduction=reduction)
    ) / 2
    loss = valid_mask.unsqueeze(1).to(loss.dtype) * loss
    if red_dim is not None:
        assert reduction == "none", "Incompatible setup."
        loss = loss.sum(dim=red_dim)
    return loss


def js_loss(
    p: Tensor,
    q: Tensor,
    ignore_index: int = -1,
    reduction: str = "mean",
) -> Tensor:
    assert_in_enum(["mean", "sum", "none"], reduction=reduction)
    loss = js_div(p, q, ignore_index=ignore_index, red_dim=(1,))
    return reduce_spatial(loss, reduction)
