"""Loss functions and LossFactory for adversarial attacks on segmentation."""

from .factory import LossFactory, LossSpec, loss_factory

__all__ = ["LossFactory", "LossSpec", "loss_factory"]
