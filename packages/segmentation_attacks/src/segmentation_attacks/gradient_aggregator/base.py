"""Base classes for gradient aggregators."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, Tuple

import torch

from ..utils.types import Tensor

if TYPE_CHECKING:
    from ..base import BaseGradientAttack


class GradientAggregator(ABC):
    r"""Base class for gradient aggregators for dense prediction tasks.

    Attributes:
        initialized: Whether the gradient aggregator is initialized.
        name: The name of the gradient aggregator.
    """

    def __init__(self, name: Optional[str] = None):
        r"""Initialize the gradient aggregator."""
        self.initialized = False
        self.name = name
        self._attack: Optional["BaseGradientAttack"] = None

    @property
    def is_initialized(self) -> bool:
        r"""Check if the gradient aggregator is initialized."""
        return self.initialized

    @property
    def model(self):
        return self._attack.model

    @property
    def loss(self):
        return self._attack.loss

    @property
    def targeted(self) -> bool:
        return self._attack.targeted

    @property
    def ignore_index(self):
        return self._attack.ignore_index

    def setup(self, attack: "BaseGradientAttack") -> "GradientAggregator":
        r"""Setup the gradient aggregator with a reference to the attack.

        Args:
            attack: The attack object. Properties (model, loss, targeted,
                ignore_index) are read from the attack on access, so changes
                to the attack are reflected automatically.

        Returns:
            GradientAggregator: The initialized gradient aggregator.
        """
        self._attack = attack
        self.initialized = True
        return self

    @abstractmethod
    def compute_gradient(
        self, x: Tensor, y: Tensor, **kwargs
    ) -> Tuple[Tensor, Tensor, Tensor]:
        r"""Compute gradient of the loss with respect to input x.

        Args:
            x: The input tensor.
            y: The target tensor.
            **kwargs: Additional keyword arguments.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: The gradient, loss, and model output.
        """
        ...


class BaseWeightedSumGradientAggregator(GradientAggregator):
    r"""Abstract class for weighted sum gradient aggregator.

    This aggregator computes the gradient of the pixel-wise weighted
    summed loss with respect to the input.
    """

    @abstractmethod
    def get_weights(self, x: Tensor, y: Tensor, **kwargs) -> Tensor:
        """Get weights for the weighted sum gradient."""
        ...

    def _compute_loss(self, logits: Tensor, y: Tensor, **kwargs) -> Tensor:
        """Compute the per-pixel loss. Override to customize loss computation.

        Args:
            logits: Model output logits.
            y: Target tensor.
            **kwargs: Additional keyword arguments forwarded to the loss.

        Returns:
            Tensor: Per-pixel loss values.
        """
        return self.loss(logits, y, **kwargs)

    def compute_gradient(
        self, x: Tensor, y: Tensor, **kwargs
    ) -> Tuple[Tensor, Tensor, Tensor]:
        assert self.is_initialized, (
            "Gradient aggregator is not initialized. Call setup() first."
        )

        def scaled_loss_fn(x_input: Tensor) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
            logits = self.model(x_input)
            loss = self._compute_loss(logits, y, **kwargs)
            if self.targeted:
                loss = -loss
            scale = self.get_weights(
                x_input, y, logits=logits.detach(), loss=loss.detach(), **kwargs
            )
            batch_loss = loss.view(x_input.shape[0], -1).mean(1)
            scaled_batch_loss = (loss * scale).view(x_input.shape[0], -1).mean(1)
            return scaled_batch_loss.sum(), (batch_loss.detach(), logits.detach())

        gradient, (batch_loss, logits) = torch.func.grad(scaled_loss_fn, has_aux=True)(
            x
        )
        return gradient, batch_loss, logits
