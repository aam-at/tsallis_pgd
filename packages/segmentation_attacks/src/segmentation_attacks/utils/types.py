"""Common shared typing primitives used across the library."""

from __future__ import annotations

from typing import Any, Protocol, TypeAlias, runtime_checkable

import torch
from torch import nn

Tensor: TypeAlias = torch.Tensor
TensorShape: TypeAlias = torch.Size
Model: TypeAlias = nn.Module


@runtime_checkable
class Loss(Protocol):
    """Callable loss function returning per-sample or per-pixel tensor values."""

    def __call__(
        self, logits: Tensor, labels: Tensor, *args: Any, **kwargs: Any
    ) -> Tensor: ...
