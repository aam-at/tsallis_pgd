"""Tsallis cross-entropy, adaptive Tsallis CE, and Tsallis JS divergence."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from ..debug import (
    assert_greater,
    assert_greater_equal,
    assert_in_enum,
    assert_in_range,
)
from .common import get_safe_targets, reduce_spatial, spatial_one_hot
from .divergence import js_div


def _tsallis_relative_entropy(
    probs: Tensor,
    reference: Tensor,
    q_param: float,
    eps: float,
) -> Tensor:
    """Tsallis q-relative entropy D_q(probs || reference)."""
    probs = probs.clamp_min(eps)
    reference = reference.clamp_min(eps)
    flat_probs = probs.view(probs.shape[0], probs.shape[1], -1)
    flat_reference = reference.view(reference.shape[0], reference.shape[1], -1)
    if math.isclose(q_param, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        rel = (flat_probs * (flat_probs.log() - flat_reference.log())).sum(dim=1)
    else:
        sum_term = (flat_probs.pow(q_param) * flat_reference.pow(1.0 - q_param)).sum(
            dim=1
        )
        rel = (1.0 - sum_term) / (q_param - 1.0)
    return rel.view(probs.shape[0], *probs.shape[2:])


def tsallis_js_div(
    p: Tensor,
    q: Tensor,
    q_param: Tensor | float = 1.2,
    *,
    softmax_output: bool = False,
    ignore_index: int = -1,
    reduction: str = "none",
    eps: float = 1e-12,
) -> Tensor:
    """Tsallis-q Jensen-Shannon divergence."""
    assert_in_enum(["none", "mean", "sum"], reduction=reduction)
    if q_param <= 0:
        raise ValueError(f"q_param must be positive, got {q_param}.")

    if math.isclose(q_param, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        loss_map = js_div(
            p,
            q,
            softmax_output=softmax_output,
            ignore_index=ignore_index,
            red_dim=(1,),
        )
        if reduction == "none":
            return loss_map
        flat = loss_map.view(loss_map.shape[0], -1)
        return flat.sum(dim=1) if reduction == "sum" else flat.mean(dim=1)

    if not softmax_output:
        probs = F.softmax(p, dim=1)
    else:
        probs = p
    probs = probs.clamp_min(eps)

    batch_size, num_classes = probs.shape[:2]
    targets = q.long()
    safe_targets, mask = get_safe_targets(targets, ignore_index)

    target_one_hot = spatial_one_hot(safe_targets, num_classes).to(probs.dtype)
    mask_expanded = mask.unsqueeze(1)
    target_probs = torch.where(mask_expanded, target_one_hot, probs)

    mixture = 0.5 * (probs + target_probs)
    div_pred = _tsallis_relative_entropy(probs, mixture, q_param, eps)
    div_target = _tsallis_relative_entropy(target_probs, mixture, q_param, eps)
    loss = 0.5 * (div_pred + div_target)
    loss = loss * mask.to(loss.dtype)

    return reduce_spatial(loss, reduction)


def tsallis_js_loss(
    p: Tensor,
    q: Tensor,
    q_param: Tensor | float = 1.2,
    *,
    softmax_output: bool = False,
    ignore_index: int = -1,
    reduction: str = "mean",
    eps: float = 1e-12,
) -> Tensor:
    return tsallis_js_div(
        p,
        q,
        q_param=q_param,
        softmax_output=softmax_output,
        ignore_index=ignore_index,
        reduction=reduction,
        eps=eps,
    )


def tsallis_ce(
    p: Tensor,
    q: Tensor,
    q_param: float | Sequence[float] | Tensor = 0.8,
    weights: Tensor | Sequence[float] | None = None,
    softmax_output: bool = False,
    ignore_index: int = -1,
    red_dim: int | tuple[int, ...] | None = None,
    reduction: str = "none",
    eps: float = 1e-12,
) -> Tensor:
    """Tsallis q-cross-entropy between predictions and labels."""
    assert_in_enum(["none", "mean", "sum"], reduction=reduction)
    if red_dim is not None and reduction != "none":
        raise ValueError("red_dim can only be used when reduction='none'.")

    if not softmax_output:
        p = F.softmax(p, dim=1)
    p = p.clamp_min(eps)

    batch_size, num_classes = p.shape[:2]
    spatial_shape = p.shape[2:]

    targets = q.long()
    targets_flat = targets.view(batch_size, -1)
    mask_flat = targets_flat != ignore_index
    safe_targets = torch.where(mask_flat, targets_flat, torch.zeros_like(targets_flat))

    probs_flat = p.reshape(batch_size, num_classes, -1)
    true_probs = torch.gather(probs_flat, 1, safe_targets.unsqueeze(1)).squeeze(1)
    true_probs = true_probs.clamp_min(eps)

    q_tensor = torch.as_tensor(q_param, dtype=p.dtype, device=p.device)
    if q_tensor.numel() == 1:
        q_tensor = q_tensor.view(1).expand(batch_size)
    else:
        if q_tensor.shape[0] != batch_size:
            raise ValueError(
                f"q_param must have shape [B] (or broadcastable), got {q_tensor.shape} "
                f"for batch size {batch_size}."
            )
        if q_tensor.ndim > 1:
            q_tensor = q_tensor.reshape(batch_size, -1)
            if q_tensor.shape[1] != 1:
                raise ValueError(
                    "q_param must be scalar or have trailing singleton dims when "
                    "providing per-sample values."
                )
            q_tensor = q_tensor[:, 0]

    q_tensor = q_tensor.view(batch_size, 1)
    is_q_one = torch.isclose(q_tensor, torch.ones_like(q_tensor), atol=1e-6, rtol=0.0)
    one_minus_q = 1.0 - q_tensor
    safe_one_minus_q = torch.where(is_q_one, torch.ones_like(one_minus_q), one_minus_q)
    ce_loss = -torch.log(true_probs)
    tsallis_loss_val = (1.0 - true_probs.pow(one_minus_q)) / safe_one_minus_q
    loss_flat = torch.where(is_q_one.expand_as(ce_loss), ce_loss, tsallis_loss_val)
    loss_flat = torch.where(mask_flat, loss_flat, torch.zeros_like(loss_flat))

    if weights is not None:
        weight_tensor = torch.as_tensor(weights, dtype=p.dtype, device=p.device)
        if weight_tensor.numel() != num_classes:
            raise ValueError(
                f"weights must have length {num_classes}, got {weight_tensor.numel()}."
            )
        weight_flat = weight_tensor[safe_targets]
        weight_flat = torch.where(
            mask_flat,
            weight_flat,
            torch.zeros_like(weight_flat, dtype=p.dtype),
        )
        loss_flat = loss_flat * weight_flat

    loss = loss_flat.view(batch_size, *spatial_shape)
    mask = mask_flat.view(batch_size, *spatial_shape).to(loss.dtype)

    if red_dim is not None:
        dims = (red_dim,) if isinstance(red_dim, int) else tuple(red_dim)
        return loss.sum(dim=dims)
    if reduction == "none":
        return loss
    loss_flat = loss.view(batch_size, -1)
    mask_flat = mask.view(batch_size, -1)
    if reduction == "sum":
        return loss_flat.sum(dim=1)
    denom = mask_flat.sum(dim=1).clamp_min(1.0)
    return loss_flat.sum(dim=1) / denom


def tsallis_ce_loss(
    p: Tensor,
    q: Tensor,
    q_param: Tensor | float = 0.8,
    ignore_index: int = -1,
    reduction: str = "mean",
) -> Tensor:
    assert_in_enum(["mean", "sum", "none"], reduction=reduction)
    loss = tsallis_ce(
        p, q, q_param=q_param, ignore_index=ignore_index, reduction="none"
    )
    _, mask = get_safe_targets(q, ignore_index)
    return reduce_spatial(loss, reduction, valid_mask=mask)


def _probability_to_q(prob: Tensor | float) -> Tensor | float:
    """Convert peak probability p* to its equivalent Tsallis q parameter."""
    return (1.0 - 2.0 * prob) / (1.0 - prob + 1e-8)


def _q_to_probability(q: Tensor | float) -> Tensor | float:
    """Convert Tsallis q parameter back to peak probability p*."""
    return (1.0 - q) / (2.0 - q + 1e-8)


def _get_param(params: Mapping[str, Any] | Any, key: str) -> Any:
    if isinstance(params, Mapping) and key in params:
        return params[key]
    if hasattr(params, key):
        return getattr(params, key)
    raise KeyError(f"Missing adaptive Tsallis parameter: {key}")


def adaptive_tsallis_ce_loss(
    logits: Tensor,
    labels: Tensor,
    params: Mapping[str, Any] | Any,
    *,
    step: int,
    iterations: int,
    ignore_index: int = -1,
    reduction: str = "mean",
    q_param: Tensor | None = None,
) -> Tensor:
    """Adaptive Tsallis cross-entropy with scheduling.

    Args:
        logits: Model output logits
        labels: Target labels
        params: Parameters for scheduling
        step: Current iteration step
        iterations: Total iterations
        ignore_index: Index to ignore in loss computation
        reduction: Loss reduction method
        q_param: Optional pre-computed q parameter (overrides scheduling if provided)
    """
    # If q_param is provided externally (e.g., from EMA tracking), use it directly
    if q_param is not None:
        q = q_param
    else:
        # Otherwise, compute q using the scheduling logic
        variant = _get_param(params, "variant")
        assert_in_enum(
            ["q_linear", "q_cosine", "p_linear", "p_cosine"], variant=variant
        )
        if not torch.compiler.is_compiling():
            assert_greater(0, iterations=iterations)
            assert_greater_equal(0, step=step)

        device, dtype = logits.device, logits.dtype
        progress = torch.clamp(
            torch.as_tensor(step, dtype=dtype, device=device)
            / torch.as_tensor(iterations, dtype=dtype, device=device),
            0.0,
            1.0,
        )

        def schedule_linear_or_cosine(
            start: float | Tensor,
            end: float | Tensor,
            end_fraction: float,
            mode: str,
        ) -> Tensor:
            if end_fraction <= 0.0 or end_fraction >= 1.0:
                scaled = progress
            else:
                freeze_start = 1.0 - end_fraction
                scaled = (progress - freeze_start) / end_fraction
                scaled = scaled.clamp(0.0, 1.0)
            if mode == "linear":
                return start + (end - start) * scaled
            if mode == "cosine":
                return end + (start - end) * (1 + torch.cos(math.pi * scaled)) / 2
            raise ValueError(f"Unknown schedule mode '{mode}'")

        if variant in ["q_linear", "q_cosine"]:
            q_start = _get_param(params, "q_start")
            q_end = _get_param(params, "q_end")
            end_fraction = _get_param(params, "end_fraction")
            assert_in_range(0.0, 1.0, end_fraction=end_fraction)
            mode = "linear" if variant == "q_linear" else "cosine"
            q = schedule_linear_or_cosine(q_start, q_end, end_fraction, mode)
        elif variant in ["p_linear", "p_cosine"]:
            p_start = _get_param(params, "p_start")
            p_end = _get_param(params, "p_end")
            end_fraction = _get_param(params, "end_fraction")
            assert_in_range(0.0, 1.0, end_fraction=end_fraction)
            mode = "linear" if variant == "p_linear" else "cosine"
            p_q = schedule_linear_or_cosine(p_start, p_end, end_fraction, mode)
            q = _probability_to_q(p_q)
        else:
            raise ValueError(f"Unsupported adaptive Tsallis variant '{variant}'.")

    return tsallis_ce_loss(
        logits,
        labels,
        q_param=q,
        ignore_index=ignore_index,
        reduction=reduction,
    )
