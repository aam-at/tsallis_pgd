from __future__ import annotations

from collections import namedtuple
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .base import DatasetInfo
from .configs import DatasetInfo, VOC2012Info
from .mixins import (
    ClassInfoMixin,
    CV2SegmentationLoaderMixin,
    PILSegmentationLoaderMixin,
    SegmentationLoaderMixin,
)


def get_voc_weights(num_classes: int, variant: str = "croce"):
    """Get VOC2012 weights for the specified number of classes."""
    if num_classes != 21:
        raise ValueError("VOC2012 weights are only available for 21 classes.")
    if variant == "croce":
        # Croce computes weights using validation set on centered crop
        # ref: Croce, F., Singh, N. D., & Hein, M. (2025). Towards reliable
        # evaluation and fast training of robust semantic segmentation models.
        VOC_WTS = [
            0.0007,
            0.0531,
            0.1394,
            0.0500,
            0.0814,
            0.0575,
            0.0256,
            0.0312,
            0.0198,
            0.0626,
            0.0382,
            0.0457,
            0.0212,
            0.0404,
            0.0421,
            0.0089,
            0.0915,
            0.0585,
            0.0366,
            0.0279,
            0.0677,
        ]
    else:
        # Weights computed on training dataset
        VOC_WTS = [
            0.0006973303388804197,
            0.0531371533870697,
            0.0588422454893589,
            0.054959218949079514,
            0.07276342809200287,
            0.0938061997294426,
            0.03987886384129524,
            0.024508647620677948,
            0.014981845393776894,
            0.03718229755759239,
            0.08789754658937454,
            0.046204112470149994,
            0.016793247312307358,
            0.05360450968146324,
            0.04128425568342209,
            0.006290586665272713,
            0.07775496691465378,
            0.07969097793102264,
            0.0404338575899601,
            0.03687003627419472,
            0.06241872161626816,
        ]
    return VOC_WTS


class _BaseVOCSegmentation(ClassInfoMixin, SegmentationLoaderMixin, Dataset):
    """Pascal VOC segmentation dataset with optional CV2/PIL backends."""

    _SPLITS_DIR = "Segmentation"
    _TARGET_DIR = "SegmentationClass"
    _TARGET_FILE_EXT = ".png"

    VOCClass = namedtuple("VOCClass", ["name", "id", "color"])

    classes = [
        VOCClass("background", 0, [0, 0, 0]),
        VOCClass("airplane", 1, [128, 0, 0]),
        VOCClass("bicycle", 2, [0, 128, 0]),
        VOCClass("bird", 3, [128, 128, 0]),
        VOCClass("boat", 4, [0, 0, 128]),
        VOCClass("bottle", 5, [128, 0, 128]),
        VOCClass("bus", 6, [0, 128, 128]),
        VOCClass("car", 7, [128, 128, 128]),
        VOCClass("cat", 8, [64, 0, 0]),
        VOCClass("chair", 9, [192, 0, 0]),
        VOCClass("cow", 10, [64, 128, 0]),
        VOCClass("diningtable", 11, [192, 128, 0]),
        VOCClass("dog", 12, [64, 0, 128]),
        VOCClass("horse", 13, [192, 0, 128]),
        VOCClass("motorcycle", 14, [64, 128, 128]),
        VOCClass("person", 15, [192, 128, 128]),
        VOCClass("potted-plant", 16, [0, 64, 0]),
        VOCClass("sheep", 17, [128, 64, 0]),
        VOCClass("sofa", 18, [0, 192, 0]),
        VOCClass("train", 19, [128, 192, 0]),
        VOCClass("tv", 20, [0, 64, 128]),
    ]

    info: DatasetInfo = VOC2012Info()

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        year: str = "2012",
        transforms=None,
        croce_labels: bool = False,
        croce_weights: bool = False,
        target_size: Tuple[int, int] | None = None,
        **kwargs,
    ):
        self.root = Path(root)
        self.split = split
        self.year = year
        self.transforms = transforms
        self.croce_labels = croce_labels
        self.croce_weights = croce_weights
        self.target_size = target_size
        self.images, self.targets = self._build_index(split=split)
        self.masks = self.targets
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected Pascal VOC dataset kwargs: {unexpected}")

    def _resolve_voc_root(self) -> Path:
        candidates = [
            self.root / "VOCdevkit" / f"VOC{self.year}",
            self.root / f"VOC{self.year}",
            self.root,
        ]
        for candidate in candidates:
            if (candidate / "JPEGImages").is_dir() and (
                candidate / "ImageSets" / self._SPLITS_DIR
            ).is_dir():
                return candidate
        return candidates[0]

    def _build_index(self, split: str) -> tuple[list[str], list[str]]:
        voc_root = self._resolve_voc_root()
        split_file = voc_root / "ImageSets" / self._SPLITS_DIR / f"{split}.txt"
        if not split_file.is_file():
            raise FileNotFoundError(
                f"Pascal VOC split file not found: {split_file}. "
                f"Set `data.dataset.root` to a valid VOC root."
            )

        file_names = [
            line.strip() for line in split_file.read_text().splitlines() if line.strip()
        ]
        image_dir = voc_root / "JPEGImages"
        target_dir = voc_root / self._TARGET_DIR
        images = [str(image_dir / f"{name}.jpg") for name in file_names]
        targets = [
            str(target_dir / f"{name}{self._TARGET_FILE_EXT}") for name in file_names
        ]
        return images, targets

    def __len__(self) -> int:
        return len(self.images)

    def sort(self):
        # do nothing
        return

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image, target = self._load_input(index)
        image, target = self.transforms(image, target)
        if self.croce_labels:
            target = self._remap_ignore_to_background(target)
        return image, target

    @staticmethod
    def _remap_ignore_to_background(target: Any) -> Any:
        if isinstance(target, torch.Tensor):
            remapped = target.clone()
            remapped[remapped == 255] = 0
            return remapped
        if isinstance(target, Image.Image):
            arr = np.array(target, copy=True)
            arr[arr == 255] = 0
            return Image.fromarray(arr)
        arr = np.array(target, copy=True)
        arr[arr == 255] = 0
        return arr

    def get_weights(self) -> list[float]:
        variant = "croce" if self.croce_weights else "normal"
        return get_voc_weights(self.info.num_classes, variant=variant)


class CV2VOCSegmentation(CV2SegmentationLoaderMixin, _BaseVOCSegmentation):
    def __init__(
        self,
        root: str | Path,
        transforms=None,
        target_size: Tuple[int, int] | None = None,
        float_images: bool = False,
        **kwargs,
    ):
        super().__init__(root, transforms=transforms, target_size=target_size, **kwargs)
        self.float_images = float_images


class PILVOCSegmentation(PILSegmentationLoaderMixin, _BaseVOCSegmentation): ...


class FFCVVOCSegmentation(CV2SegmentationLoaderMixin, _BaseVOCSegmentation):
    """Pascal VOC dataset for creating FFCV format files.

    This class is used during FFCV file creation (prepare_data phase).
    For actual training with FFCV, use FFCVSegmentationDataModule(...).
    """

    def __init__(
        self,
        root: str | Path,
        transforms=None,
        target_size: Tuple[int, int] | None = None,
        ffcv_path: str | Path | None = None,
        **kwargs,
    ):
        super().__init__(root, transforms=transforms, target_size=target_size, **kwargs)
        self.ffcv_path = ffcv_path
