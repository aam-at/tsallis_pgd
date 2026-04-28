"""Wrapper for mmsegmentation models loaded from config + checkpoint."""

import os
from pathlib import Path
from typing import Tuple

import torch
import yaml
from torch import Tensor

from .wrapper import SegmentationModelWrapper


def _find_mmseg_package_config(config_name: str) -> str | None:
    """Find a config by basename inside the installed mmseg package."""
    import mmseg

    mim_root = Path(mmseg.__file__).resolve().parent / ".mim" / "configs"
    matches = list(mim_root.rglob(config_name))
    if matches:
        return str(matches[0])
    return None


def _resolve_mmseg_config_path(config: str) -> str:
    """Resolve an mmseg config path, preferring installed `.mim/configs`."""
    config_name = Path(config).name
    package_config = _find_mmseg_package_config(config_name)
    if package_config is not None:
        return package_config
    if os.path.isfile(config):
        return config
    return config


def _resolve_mmseg_checkpoint_path(config: str, checkpoint: str | None) -> str | None:
    """Resolve a checkpoint from a local file, URL, or mmseg metafile entry."""
    if checkpoint is None or os.path.isfile(checkpoint):
        return checkpoint
    if checkpoint.startswith(("http://", "https://")):
        return checkpoint

    import mmseg

    mim_root = Path(mmseg.__file__).resolve().parent / ".mim" / "configs"
    config_name = Path(config).name
    checkpoint_name = Path(checkpoint).name

    for metafile in mim_root.rglob("metafile.y*ml"):
        with metafile.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        for model in data.get("Models", []):
            model_config = model.get("Config", "")
            weights = model.get("Weights")
            if not weights:
                continue
            if (
                Path(model_config).name == config_name
                or Path(weights).name == checkpoint_name
            ):
                return weights

    return checkpoint


class MMSegModel(SegmentationModelWrapper):
    """Wrapper for mmsegmentation models initialised from config + checkpoint."""

    def __init__(
        self,
        config: str,
        checkpoint: str,
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
        from mmseg.apis import init_model

        resolved_config = _resolve_mmseg_config_path(config)
        resolved_checkpoint = _resolve_mmseg_checkpoint_path(
            resolved_config, checkpoint
        )
        name = name or os.path.splitext(os.path.basename(resolved_config))[0]

        model = init_model(resolved_config, resolved_checkpoint)

        super().__init__(
            model=model,
            num_classes_attr="decode_head.num_classes",
            mean=mean,
            std=std,
            normalize=normalize,
            compile=compile,
            freeze=freeze,
            name=name,
            device=device,
            input_min=input_min,
            input_max=input_max,
        )

    def forward(self, x: Tensor, sanity_check: bool = True) -> Tensor:
        if sanity_check and not torch.compiler.is_compiling():
            self._sanity_check_input(x)
        if self.normalize:
            x = (x - self.mean) / self.std
        img_metas = {
            "ori_shape": x.shape[-2:] + (x.shape[1],),
            "img_shape": x.shape[-2:],
        }
        return self.model.inference(x, batch_img_metas=[img_metas])
