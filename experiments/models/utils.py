from collections import OrderedDict

import torch
from torch import Tensor, nn


def requires_grad_(module: nn.Module, requires_grad: bool) -> nn.Module:
    for parameter in module.parameters():
        parameter.requires_grad_(requires_grad)
    return module


class _InnerAttrAccessMixin:
    """Mixin that delegates attribute access to inner submodules."""

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            for module in self._modules.values():
                try:
                    return getattr(module, name)
                except AttributeError:
                    pass
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )


class SequentialWrapper(_InnerAttrAccessMixin, nn.Sequential):
    """nn.Sequential that delegates attribute access to inner modules."""


class DataparallelWrapper(_InnerAttrAccessMixin, nn.DataParallel):
    """nn.DataParallel that delegates attribute access to inner modules."""


class ImageNormalizer(nn.Module):
    def __init__(
        self, mean: tuple[float, float, float], std: tuple[float, float, float]
    ) -> None:
        super().__init__()

        self.register_buffer(
            "mean", torch.as_tensor(mean).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "std", torch.as_tensor(std).view(1, 3, 1, 1), persistent=False
        )

    def forward(self, input: Tensor) -> Tensor:
        return (input - self.mean) / self.std


def normalize_model(
    model: nn.Module,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> nn.Module:
    layers = OrderedDict([("normalize", ImageNormalizer(mean, std)), ("model", model)])
    return SequentialWrapper(layers)
