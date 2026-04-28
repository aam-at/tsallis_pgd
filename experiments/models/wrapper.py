"""Unified wrapper for standard PyTorch segmentation models."""

from typing import Tuple

import torch
from torch import Tensor, nn

from .utils import requires_grad_


class SegmentationModelWrapper(nn.Module):
    """Wrapper for standard PyTorch segmentation models.

    Handles checkpoint loading, normalization, input validation,
    optional compilation and freezing.
    """

    def __init__(
        self,
        model: nn.Module,
        checkpoint: str | None = None,
        num_classes_attr: str | None = None,
        checkpoint_key: str | None = None,
        checkpoint_prefix: str | None = None,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        normalize: bool = True,
        compile: bool = False,
        freeze: bool = True,
        name: str | None = None,
        device: str = "cuda",
        input_min: float = 0.0,
        input_max: float = 1.0,
    ):
        super().__init__()
        self.device = device
        self.name = name
        self.num_classes_attr = num_classes_attr or getattr(
            type(self), "num_classes_attr", None
        )
        self.register_buffer("mean", torch.as_tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.as_tensor(std).view(1, 3, 1, 1))
        self.normalize = normalize
        self.input_min = input_min
        self.input_max = input_max

        if checkpoint is not None:
            self._load_checkpoint(
                model=model,
                checkpoint=checkpoint,
                checkpoint_key=checkpoint_key,
                checkpoint_prefix=checkpoint_prefix,
            )

        self.model = model
        self.to(device)
        if compile:
            self.model = torch.compile(self.model)
        if freeze:
            requires_grad_(self.model, False)

    def _load_checkpoint(
        self,
        model: nn.Module,
        checkpoint: str,
        checkpoint_key: str | None = None,
        checkpoint_prefix: str | None = None,
    ) -> None:
        ckpt = torch.load(checkpoint, map_location="cpu")
        if checkpoint_key is not None:
            ckpt = ckpt[checkpoint_key]
        if checkpoint_prefix is not None:
            ckpt = {
                (
                    key[len(checkpoint_prefix) :]
                    if key.startswith(checkpoint_prefix)
                    else key
                ): value
                for key, value in ckpt.items()
            }
        model.load_state_dict(ckpt, strict=True)

    @property
    def num_classes(self) -> int:
        if self.num_classes_attr is None:
            raise AttributeError("`num_classes_attr` is not configured for this model.")
        obj = self.model
        for attr in self.num_classes_attr.split("."):
            obj = getattr(obj, attr)
        return obj

    def _sanity_check_input(self, x: Tensor) -> None:
        x_min = x.min().item()
        x_max = x.max().item()
        assert x_min >= self.input_min and x_max <= self.input_max, (
            f"Input out of bounds: expected [{self.input_min}, {self.input_max}], "
            f"got [{x_min}, {x_max}]"
        )

    def forward(self, x: Tensor, sanity_check: bool = True) -> Tensor:
        if sanity_check and not torch.compiler.is_compiling():
            self._sanity_check_input(x)
        if self.normalize:
            x = (x - self.mean) / self.std
        return self.model(x)
