"""Stateless functional operations: projections, norms, and the vectorize decorator."""

from __future__ import annotations

import functools
import inspect
import types
import typing

import torch

from .debug import assert_greater, assert_rank
from .types import Tensor


def l2_normalize(d: Tensor, eps: float = 1e-12) -> Tensor:
    """Stable L2 normalization."""
    eps_t = torch.as_tensor(eps, dtype=d.dtype, device=d.device)
    shp = d.shape
    d = d.flatten(1)
    d = d / (eps_t + torch.max(torch.abs(d), dim=1, keepdim=True)[0])
    d_square_sum = torch.sum(d**2, dim=1, keepdim=True)
    d_inv_norm = (eps_t + d_square_sum).rsqrt()
    return (d * d_inv_norm).reshape(shp)


def linf_normalize(d: Tensor) -> Tensor:
    """Linf normalization (sign function)."""
    return torch.sign(d)


def optimal_linear_l2(
    v: Tensor,
    ε: Tensor,
    eps: float = 1e-12,
    sanity_check: bool = False,
) -> Tensor:
    """Optimal input to a linear function under L2-norm constraint."""
    if sanity_check:
        assert_rank(2, v=v)
        assert_rank(1, ε=ε)
        assert_greater(ε=ε, zero=0)
    return ε.unsqueeze(1) * l2_normalize(v, eps=eps)


def optimal_linear_linf(
    v: Tensor,
    ε: Tensor,
    sanity_check: bool = False,
) -> Tensor:
    """Optimal input to a linear function under Linf-norm constraint."""
    if sanity_check:
        assert_rank(2, v=v)
        assert_rank(1, ε=ε)
        assert_greater(ε=ε, zero=0)
    return ε.unsqueeze(1) * linf_normalize(v)


def project_l2(
    v: Tensor,
    ε: Tensor,
    sanity_check: bool = False,
) -> Tensor:
    """Project *v* onto the L2 ball of radius *ε*."""
    if sanity_check:
        assert_rank(2, v=v)
        assert_rank(1, ε=ε)
        assert_greater(ε=ε, zero=0)
    n = torch.norm(v, p=2, dim=1)
    return v / torch.max(torch.tensor(1.0), n.unsqueeze(-1) / ε.unsqueeze(-1))


def project_linf(
    v: Tensor,
    ε: Tensor,
    sanity_check: bool = False,
) -> Tensor:
    """Project *v* onto the Linf ball of radius *ε*."""
    if sanity_check:
        assert_rank(2, v=v)
        assert_rank(1, ε=ε)
        assert_greater(ε=ε, zero=0)
    ε_exp = ε.view(-1, *[1] * (v.ndim - 1))
    return torch.clamp(v, -ε_exp, ε_exp)


def vectorize(fn, reshape_output: bool = True):
    """Decorator that flattens spatial dims for functions operating on [B, D] tensors."""
    arg_names = inspect.getfullargspec(fn).args
    # Resolve string annotations from `from __future__ import annotations`.
    # Match args annotated as Tensor or Tensor | None (union with None).
    try:
        resolved = typing.get_type_hints(fn)
    except Exception:
        resolved = inspect.getfullargspec(fn).annotations

    def _is_tensor_type(tp: type) -> bool:
        if tp is Tensor:
            return True
        if isinstance(tp, types.UnionType):
            return Tensor in tp.__args__
        origin = typing.get_origin(tp)
        if origin is typing.Union:
            return Tensor in typing.get_args(tp)
        return False

    tensor_args = {k for k, v in resolved.items() if _is_tensor_type(v)}

    @functools.wraps(fn)
    def wrapper(*arg_values, **kwargs):
        new_args = []
        shp_set = False
        shp_static = None
        shp_dynamic = None
        for arg_name, arg_value in zip(arg_names, arg_values):
            if (
                arg_name in tensor_args
                and isinstance(arg_value, torch.Tensor)
                and arg_value.dim() > 2
            ):
                if not shp_set:
                    shp_static = arg_value.size()
                    shp_dynamic = arg_value.shape
                    shp_set = True
                else:
                    assert arg_value.dim() == len(shp_static)
                new_arg = arg_value.reshape(shp_dynamic[0], -1)
                new_args.append(new_arg)
            else:
                new_args.append(arg_value)
        result = fn(*new_args, **kwargs)
        if reshape_output and shp_set:
            if isinstance(result, (tuple, list)):
                return [
                    x_.reshape(shp_dynamic) if x_.dim() == 2 else x_ for x_ in result
                ]
            else:
                return result.reshape(shp_dynamic)
        return result

    return wrapper
