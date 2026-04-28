"""Cosine similarity weighted gradient aggregator (CosPGD)."""

import torch
import torch.nn.functional as F

from ..utils.types import Tensor
from .base import BaseWeightedSumGradientAggregator


class CosineSimilarityWeightedSumGradientAggregator(BaseWeightedSumGradientAggregator):
    """Gradient aggregator based on cosine similarity between prediction and
    target based on CosPGD paper.

    Reference:
    - Agnihotri, S., Jung, S., & Keuper, M. (2024). CosPGD: an efficient
      white-box adversarial attack for pixel-wise prediction tasks. In R.
      Salakhutdinov, Z. Kolter, K. Heller, A. Weller, N. Oliver, J. Scarlett, &
      F. Berkenkamp, Proceedings of the 41st International Conference on
      Machine Learning (pp. 416–451). : PMLR.
    """

    def __init__(self, discrete_mode: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.discrete_mode = discrete_mode

    def get_weights(
        self, x: Tensor, y: Tensor, logits: Tensor | None = None, **kwargs
    ) -> Tensor:
        assert logits is not None, (
            "Logits must be provided to compute cosine similarity"
        )
        # compute cosine similarity between the prediction and the target
        num_classes = self.model.num_classes
        if self.discrete_mode:
            y_clamped = torch.clamp(y, 0, num_classes - 1).long()
            targets = torch.zeros(
                *y_clamped.shape, num_classes, dtype=logits.dtype, device=logits.device
            ).scatter(-1, y_clamped.unsqueeze(-1), 1.0)
            targets = targets.movedim(-1, 1)
        else:
            targets = F.softmax(y, dim=1)
        pred_probs = F.softmax(logits, dim=1)
        cosine_similarity = F.cosine_similarity(pred_probs, targets, dim=1)
        if self.targeted:
            cosine_similarity = 1 - cosine_similarity
        return cosine_similarity
