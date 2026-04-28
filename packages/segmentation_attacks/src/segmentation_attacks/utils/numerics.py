"""Internal utilities: assertions, tensor operations, numeric helpers."""

from __future__ import annotations

import torch

from .types import Tensor


def is_close(
    a: Tensor,
    b: Tensor | float | int,
    symmetric: bool = True,
    atol: float = 1e-8,
    rtol: float = 0.0,
) -> Tensor:
    """Element-wise closeness check."""
    b = torch.as_tensor(b, dtype=a.dtype, device=a.device)
    return _close(a, b, atol=atol, rtol=rtol)


def is_less(a: Tensor, b: Tensor, atol: float = 1e-8, rtol: float = 0.0):
    return torch.logical_and(
        torch.logical_or(_less(a, b, atol, rtol), _greater(b, a, atol, rtol)),
        torch.logical_not(_close(a, b, atol, rtol)),
    )


def is_less_equal(a: Tensor, b: Tensor, atol: float = 1e-8, rtol: float = 0.0):
    return torch.logical_not(is_greater(a, b, atol, rtol))


def is_greater(a: Tensor, b: Tensor, atol: float = 1e-8, rtol: float = 0.0):
    return is_less(b, a, atol, rtol)


def is_greater_equal(a: Tensor, b: Tensor, atol: float = 1e-8, rtol: float = 0.0):
    return is_less_equal(b, a, atol, rtol)


def _close(a: Tensor, b: Tensor, atol: float = 1e-8, rtol: float = 0.0):
    return torch.isclose(a, b, atol=atol, rtol=rtol)


def _less(a: Tensor, b: Tensor, atol: float = 1e-8, rtol: float = 0.0):
    return a < atol + (1 + rtol) * b


def _greater(a: Tensor, b: Tensor, atol: float = 1e-8, rtol: float = 0.0):
    return a > atol + (1 + rtol) * b
