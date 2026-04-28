"""Norm-specific operations for adversarial attacks (foolbox-inspired)."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from .utils.numerics import is_close
from .utils.types import Tensor
from .utils.math_ops import (
    l2_normalize,
    optimal_linear_l2,
    optimal_linear_linf,
    project_l2,
    project_linf,
    vectorize,
)


class NormOps(ABC):
    """Abstract interface for norm-specific attack operations.

    Encapsulates the three norm-dependent primitives used by iterative
    gradient attacks: random perturbation, gradient normalization, and
    perturbation projection.
    """

    @abstractmethod
    def random_perturbation(self, x0: Tensor, epsilon: Tensor) -> Tensor:
        """Generate a random starting point within the norm ball."""
        ...

    @abstractmethod
    def steepest_descent(
        self,
        x0: Tensor,
        xcurr: Tensor,
        g: Tensor,
        stepsize: Tensor,
        clip_min: float,
        clip_max: float,
        tol: float = 1e-8,
    ) -> Tensor:
        """Compute the optimal step direction under the norm constraint."""
        ...

    @abstractmethod
    def project_perturbation(
        self,
        x0: Tensor,
        xcurr: Tensor,
        epsilon: Tensor,
        clip_min: float | None,
        clip_max: float | None,
    ) -> Tensor:
        """Project perturbation onto the norm ball and clip to bounds."""
        ...


class L2NormOps(NormOps):
    """L2-norm operations."""

    def random_perturbation(self, x0: Tensor, epsilon: Tensor) -> Tensor:
        r0 = (2 * torch.rand_like(x0) - 1.0).to(x0)
        r0 = epsilon.view(-1, *[1] * (x0.ndim - 1)) * l2_normalize(r0)
        return x0 + r0

    @vectorize
    def steepest_descent(
        self,
        x0: Tensor,
        xcurr: Tensor,
        g: Tensor,
        stepsize: Tensor,
        clip_min: float,
        clip_max: float,
        tol: float = 1e-8,
    ) -> Tensor:
        bad_pos = torch.logical_or(
            torch.logical_and(xcurr >= clip_max, g > 0),
            torch.logical_and(xcurr <= clip_min, g < 0),
        )
        g = torch.where(bad_pos, torch.zeros_like(g), g)
        g_norm = torch.norm(g, p=2, dim=1, keepdim=True)
        g = torch.where(is_close(g_norm, 0, atol=tol), torch.randn_like(g), g)
        return optimal_linear_l2(g, stepsize)

    @vectorize
    def project_perturbation(
        self,
        x0: Tensor,
        xcurr: Tensor,
        epsilon: Tensor,
        clip_min: float | None,
        clip_max: float | None,
    ) -> Tensor:
        if clip_min is not None and clip_max is not None:
            xcurr = torch.clamp(xcurr, clip_min, clip_max)
        r = xcurr - x0
        r = project_l2(r, epsilon)
        return x0 + r


class LinfNormOps(NormOps):
    """Linf-norm operations."""

    def random_perturbation(self, x0: Tensor, epsilon: Tensor) -> Tensor:
        r0 = (2 * torch.rand_like(x0) - 1.0).to(x0)
        r0 = r0 * epsilon.view(-1, *[1] * (x0.ndim - 1))
        return x0 + r0

    @vectorize
    def steepest_descent(
        self,
        x0: Tensor,
        xcurr: Tensor,
        g: Tensor,
        stepsize: Tensor,
        clip_min: float,
        clip_max: float,
        tol: float = 1e-8,
    ) -> Tensor:
        return optimal_linear_linf(g, stepsize)

    @vectorize
    def project_perturbation(
        self,
        x0: Tensor,
        xcurr: Tensor,
        epsilon: Tensor,
        clip_min: float | None,
        clip_max: float | None,
    ) -> Tensor:
        if clip_min is not None and clip_max is not None:
            xcurr = torch.clamp(xcurr, clip_min, clip_max)
        r = xcurr - x0
        r = project_linf(r, epsilon)
        return x0 + r
