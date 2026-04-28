"""Madry PGD attack with L2 and Linf variants."""

from __future__ import annotations

import numpy as np

from .utils.types import Model
from .base import IterativeGradientAttack
from .norms import L2NormOps, LinfNormOps


class L2PGDAttack(IterativeGradientAttack):
    """L2 Projected Gradient Descent attack.

    Reference: Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A.
    (2018), Towards deep learning models resistant to adversarial attacks, In ,
    International Conference on Learning Representations.
    """

    name: str = "pgd_l2"

    def __init__(self, model: Model, **kwargs):
        super().__init__(model, norm_ops=L2NormOps(), **kwargs)


class LinfPGDAttack(IterativeGradientAttack):
    """Linf Projected Gradient Descent attack.

    Reference: Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A.
    (2018), Towards deep learning models resistant to adversarial attacks, In ,
    International Conference on Learning Representations.
    """

    name: str = "pgd_linf"

    def __init__(self, model: Model, **kwargs):
        super().__init__(model, norm_ops=LinfNormOps(), **kwargs)


class PGDAttack:
    """Factory that returns L2PGDAttack or LinfPGDAttack based on ``ord``.

    Preserves backward compatibility with ``PGDAttack(model, ord=2, ...)``.
    """

    def __new__(cls, model: Model, ord: str | int = np.inf, **kwargs):
        if ord in (2, "2", "L2"):
            return L2PGDAttack(model, **kwargs)
        elif ord in (np.inf, "Linf"):
            return LinfPGDAttack(model, **kwargs)
        raise ValueError(f"{ord=} is not supported. Use 2, 'L2', np.inf, or 'Linf'.")
