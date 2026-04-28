"""Internal utilities: assertions, tensor operations, numeric helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import fields, is_dataclass
from typing import Any

import torch
from torch import Tensor


def tensor_detach(tensors: Any) -> Any:
    """Recursively detach all tensors in a nested dict/list/tensor/dataclass structure."""
    if isinstance(tensors, torch.Tensor):
        return tensors.detach()
    elif isinstance(tensors, dict):
        return {key: tensor_detach(value) for key, value in tensors.items()}
    elif is_dataclass(tensors) and not isinstance(tensors, type):
        return type(tensors)(
            *[tensor_detach(getattr(tensors, f.name)) for f in fields(tensors)]
        )
    elif isinstance(tensors, Sequence) and not isinstance(tensors, (str, bytes)):
        return type(tensors)([tensor_detach(t) for t in tensors])
    else:
        raise TypeError(f"Unsupported type: {type(tensors)}")


def batch_where(
    tensors: Any,
    updates: Any,
    mask: Tensor,
) -> Any:
    """Select between *updates* and *tensors* element-wise using a boolean mask.

    Works recursively on nested dicts/lists/dataclasses of tensors.  For each leaf
    tensor the mask is broadcast along non-batch dimensions via
    ``torch.where``.  No in-place mutation is performed.

    Args:
        tensors: Current values (tensor or nested dict/list/dataclass of tensors).
        updates: New values with the same structure and batch size.
        mask: 1-D boolean tensor of shape ``[B]``.
    """
    if isinstance(tensors, dict):
        return {k: batch_where(tensors[k], updates[k], mask) for k in tensors}
    if is_dataclass(tensors) and not isinstance(tensors, type):
        return type(tensors)(
            *[
                batch_where(getattr(tensors, f.name), getattr(updates, f.name), mask)
                for f in fields(tensors)
            ]
        )
    if isinstance(tensors, Sequence) and not isinstance(tensors, (str, bytes)):
        return type(tensors)(
            [batch_where(t, u, mask) for t, u in zip(tensors, updates)]
        )
    mask_exp = mask.view(-1, *([1] * (tensors.ndim - 1)))
    return torch.where(mask_exp, updates, tensors)
