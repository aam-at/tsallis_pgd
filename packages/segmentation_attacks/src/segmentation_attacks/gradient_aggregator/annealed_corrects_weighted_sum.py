"""Annealed corrects weighted gradient aggregator (SegPGD)."""

import torch

from ..utils.types import Tensor
from .base import BaseWeightedSumGradientAggregator


class AnnealedCorrectsWeightedSumGradientAggregator(BaseWeightedSumGradientAggregator):
    r"""Gradient aggregator with annealing weights for correct and wrong
    predictions based on SegPGD paper.

    Reference:
    - Gu, J., Zhao, H., Tresp, V., & Torr, P. H. S. (2022). Segpgd: an
      effective and efficient adversarial attack for evaluating and boosting
      segmentation robustness. In S. Avidan, G. Brostow, M. Ciss{\'e}, G. M.
      Farinella, & T. Hassner, Computer Vision -- ECCV 2022 (pp. 308–325).
      Cham: Springer Nature Switzerland.
    """

    def get_weights(
        self,
        x: Tensor,
        y: Tensor,
        logits: Tensor | None = None,
        iterations: int = -1,
        step: int = -1,
        **kwargs,
    ) -> Tensor:
        """Get weights for the weighted sum gradient."""
        assert logits is not None, "Logits must be provided to compute annealed weights"
        if not torch.compiler.is_compiling():
            assert iterations > 0, f"iterations must be positive, got {iterations}"
            assert step >= 0, f"step must be non-negative, got {step}"

        with torch.no_grad():
            pred = logits.argmax(dim=1)
            is_correct = (pred == y).to(logits.dtype)
            lmbd = step / (2 * iterations)
            scale = (1 - lmbd) * is_correct + lmbd * (1 - is_correct)
        return scale
