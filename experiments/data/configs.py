"""Hydra config store registration for dataset configurations.

This module registers dataset info configurations with Hydra's config store,
allowing them to be referenced in YAML configs.

Usage:
    # In your Python entry point, before hydra.main():
    from data.configs import register_configs
    register_configs()

    # Then in YAML configs, use:
    defaults:
      - data_info: voc2012  # or cityscapes, ade20k, voc_aug
"""

from __future__ import annotations

from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore

from data.base import DatasetInfo


@dataclass
class VOC2012Info(DatasetInfo):
    """Pascal VOC 2012 dataset info."""

    input_size: tuple[int, int] = (512, 512)
    input_min: float = 0.0
    input_max: float = 255.0
    num_classes: int = 21
    ignore_index: int = 255
    train_val_test_split: tuple[int, int, int] = (1464, 1449, 1456)
    mean: tuple[float, float, float] = (123.675, 116.28, 103.53)
    std: tuple[float, float, float] = (58.395, 57.12, 57.375)


@dataclass
class VOCAugInfo(DatasetInfo):
    """Pascal VOC Augmented dataset info."""

    input_size: tuple[int, int] = (512, 512)
    input_min: float = 0.0
    input_max: float = 255.0
    num_classes: int = 21
    ignore_index: int = 255
    train_val_test_split: tuple[int, int, int] = (10582, 1449, 1456)
    mean: tuple[float, float, float] = (123.675, 116.28, 103.53)
    std: tuple[float, float, float] = (58.395, 57.12, 57.375)


@dataclass
class CityscapesInfo(DatasetInfo):
    """Cityscapes dataset info."""

    input_size: tuple[int, int] = (1024, 2048)
    input_min: float = 0.0
    input_max: float = 255.0
    num_classes: int = 19
    ignore_index: int = 255
    train_val_test_split: tuple[int, int, int] = (2975, 500, 1525)
    mean: tuple[float, float, float] = (123.675, 116.28, 103.53)
    std: tuple[float, float, float] = (58.395, 57.12, 57.375)


@dataclass
class ADE20KInfo(DatasetInfo):
    """ADE20K dataset info."""

    input_size: tuple[int, int] = (512, 512)
    input_min: float = 0.0
    input_max: float = 255.0
    num_classes: int = 150
    ignore_index: int = 255
    train_val_test_split: tuple[int, int, int] = (20210, 2000, 3352)
    mean: tuple[float, float, float] = (123.675, 116.28, 103.53)
    std: tuple[float, float, float] = (58.395, 57.12, 57.375)


@dataclass
class DataInfoConfigs:
    """Container for all dataset info configs."""

    voc2012: VOC2012Info = field(default_factory=VOC2012Info)
    voc_aug: VOCAugInfo = field(default_factory=VOCAugInfo)
    cityscapes: CityscapesInfo = field(default_factory=CityscapesInfo)
    ade20k: ADE20KInfo = field(default_factory=ADE20KInfo)


def register_configs(cs: ConfigStore | None = None) -> None:
    """Register dataset info configs with Hydra config store.

    Args:
        cs: ConfigStore instance. If None, uses the global instance.
    """
    if cs is None:
        cs = ConfigStore.instance()

    cs.store(
        group="data_info",
        name="voc2012",
        node=VOC2012Info,
        package="data.info",
    )
    cs.store(
        group="data_info",
        name="voc_aug",
        node=VOCAugInfo,
        package="data.info",
    )
    cs.store(
        group="data_info",
        name="cityscapes",
        node=CityscapesInfo,
        package="data.info",
    )
    cs.store(
        group="data_info",
        name="ade20k",
        node=ADE20KInfo,
        package="data.info",
    )
