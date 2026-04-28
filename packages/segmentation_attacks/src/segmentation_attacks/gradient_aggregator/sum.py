"""Standard sum gradient aggregator."""

import torch

from ..utils.types import Tensor
from .base import BaseWeightedSumGradientAggregator


class SumGradientAggregator(BaseWeightedSumGradientAggregator):
    r"""Standard sum gradient aggregator.

    This aggregator computes the gradient of the pixel-wise summed loss
    with respect to the input. It is equivalent to the standard PGD
    attack for dense prediction tasks.
    """

    def get_weights(self, x: Tensor, y: Tensor, **kwargs) -> Tensor:
        """Get weights for the weighted sum gradient."""
        # For standard sum, weights are all ones
        return torch.ones_like(y)
