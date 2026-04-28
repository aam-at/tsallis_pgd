from __future__ import annotations

from collections import namedtuple
from pathlib import Path
from typing import Any, Sequence, Tuple

import cv2
import numpy as np
import torchvision
from cityscapesscripts.helpers.labels import labels as city_labels
from PIL import Image
from torchvision.datasets.utils import verify_str_arg

from .configs import CityscapesInfo, DatasetInfo
from .mixins import (
    ClassInfoMixin,
    CV2SegmentationLoaderMixin,
    PILSegmentationLoaderMixin,
    SegmentationLoaderMixin,
)


def get_cityscapes_weights(num_classes: int) -> list[float]:
    if num_classes != 19:
        raise ValueError("Cityscapes weights are only available for 19 classes.")
    # Weights computed on training dataset without any resize
    CITY_WTS = [
        0.0006065628840588033,
        0.003675291780382395,
        0.000980328069999814,
        0.03410545364022255,
        0.025468867272138596,
        0.018223319202661514,
        0.10728859901428223,
        0.04046547785401344,
        0.0014054407365620136,
        0.019308142364025116,
        0.005576711613684893,
        0.018377233296632767,
        0.1658867746591568,
        0.0031953658908605576,
        0.08358465880155563,
        0.095029316842556,
        0.09600316733121872,
        0.22677022218704224,
        0.0540490485727787,
    ]
    return CITY_WTS


def cityscapes_lut(ignore_index: int = 255):
    """Build a labelId -> trainId lookup table from the official Cityscapes
    label spec."""
    label_lut = [ignore_index for _ in range(256)]
    for lbl in city_labels:
        if lbl.id < 0 or lbl.trainId in (-1, 255):
            continue
        label_lut[lbl.id] = lbl.trainId
    return label_lut


class _BaseCityscapes(
    ClassInfoMixin, SegmentationLoaderMixin, torchvision.datasets.Cityscapes
):
    CityscapesClass = namedtuple("CityscapesClass", ["name", "id", "color"])
    info: DatasetInfo = CityscapesInfo()

    classes = []
    for label in city_labels:
        if label.trainId >= 0 and label.trainId != CityscapesInfo().ignore_index:
            classes.append(CityscapesClass(label.name, label.trainId, label.color))

    _VALID_TARGET_TYPES = (
        "instance",
        "semantic",
        "semantic_labelids",
        "semantic_trainids",
        "color",
    )
    _DEFAULT_TARGET_TYPE = "semantic_trainids"

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        mode: str = "fine",
        target_type: Sequence[str] | str = _DEFAULT_TARGET_TYPE,
        transforms=None,
        target_size: Tuple[int, int] | None = None,
        **kwargs,
    ) -> None:
        """Initialize the Cityscapes dataset.

        Args:
            root (str | Path): Root directory of the Cityscapes dataset.
            split (str): The image split to use, 'train', 'test' or 'val'.
            mode (str): The quality mode to use, 'gtFine' or 'gtCoarse'.
            target_type (Sequence[str] | str): The type of target to use.
            transforms (callable, optional): A function/transform that takes in
                an PIL image and returns a transformed version.
            target_size (tuple, optional): The target size of the image.
            **kwargs: Additional arguments passed to the base class.
        """
        normalized_target_type = self._normalize_target_type(target_type)
        canonical_target_type = self._canonical_target_type(normalized_target_type)
        super().__init__(
            root,
            split=split,
            mode=mode,
            target_type=canonical_target_type,
            transforms=transforms,
            **kwargs,
        )
        self.target_type = normalized_target_type
        self.transforms = transforms
        self.target_size = target_size
        # support only one target
        self.targets = [
            t[0] if isinstance(t, (list, tuple)) else t for t in self.targets
        ]
        if self.target_type == "semantic_trainids":
            self.targets = [
                t.replace("labelIds.png", "labelTrainIds.png") for t in self.targets
            ]
        self.masks = self.targets
        self._train_id_lut = np.array(
            cityscapes_lut(ignore_index=self.info.ignore_index), dtype=np.uint8
        )

    def sort(self) -> None:
        """Sort the images and targets by city and frame number.

        This ensures a deterministic order for the dataset.
        """

        def extract_sort_key(path: str) -> Tuple[str, int]:
            # Example path: .../city/city_123456_123456_leftImg8bit.png
            p = Path(path)
            parts = p.name.split("_")
            city = parts[0]
            seq_id = int(parts[1])
            return city, seq_id

        sort_indices = sorted(
            range(len(self)), key=lambda i: extract_sort_key(self.images[i])
        )
        self.images = [self.images[i] for i in sort_indices]
        self.targets = [self.targets[i] for i in sort_indices]
        self.masks = [self.masks[i] for i in sort_indices]

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item.
        """
        image, target = self._load_input(index)
        target = self._maybe_map_to_train_ids(target)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def _get_target_suffix(self, mode: str, target_type: str) -> str:
        if target_type == "instance":
            return f"{mode}_instanceIds.png"
        elif target_type in {"semantic", "semantic_labelids"}:
            return f"{mode}_labelIds.png"
        elif target_type == "semantic_trainids":
            return f"{mode}_labelTrainIds.png"
        elif target_type == "color":
            return f"{mode}_color.png"
        else:
            return f"{mode}_polygons.json"

    def _normalize_target_type(self, target_type: Sequence[str] | str) -> str:
        target_types = (
            [target_type] if isinstance(target_type, str) else list(target_type)
        )
        if len(target_types) != 1:
            raise ValueError(
                f"Cityscapes supports a single target_type; got {target_types}."
            )

        normalized_target_type = target_types[0]
        verify_str_arg(normalized_target_type, "target_type", self._VALID_TARGET_TYPES)
        return normalized_target_type

    def _canonical_target_type(self, target_type: str) -> list[str]:
        if "semantic" in target_type:
            return ["semantic"]
        return [target_type]

    def _maybe_map_to_train_ids(self, target: Any) -> Any:
        if self.target_type == "semantic_trainids":
            return target

        if isinstance(target, Image.Image):
            return target.point(self._train_id_lut)

        if isinstance(target, np.ndarray):
            return cv2.LUT(target.astype(np.uint8), self._train_id_lut)

        return target

    def get_weights(self) -> list[float]:
        """Get class weights for the dataset."""
        return get_cityscapes_weights(self.info.num_classes)


class CV2Cityscapes(CV2SegmentationLoaderMixin, _BaseCityscapes):
    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        mode: str = "fine",
        target_type: Sequence[str] | str = _BaseCityscapes._DEFAULT_TARGET_TYPE,
        transforms=None,
        target_size: Tuple[int, int] | None = None,
        float_images: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            root=root,
            split=split,
            mode=mode,
            target_type=target_type,
            transforms=transforms,
            target_size=target_size,
            **kwargs,
        )
        self.float_images = float_images


class PILCityscapes(PILSegmentationLoaderMixin, _BaseCityscapes):
    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        mode: str = "fine",
        target_type: Sequence[str] | str = _BaseCityscapes._DEFAULT_TARGET_TYPE,
        transforms=None,
        target_size: Tuple[int, int] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            root=root,
            split=split,
            mode=mode,
            target_type=target_type,
            transforms=transforms,
            target_size=target_size,
            **kwargs,
        )


class FFCVCityscapes(CV2SegmentationLoaderMixin, _BaseCityscapes):
    """Cityscapes dataset for creating FFCV format files.

    This class is used during FFCV file creation (prepare_data phase).
    For actual training with FFCV, use SegmentationDataModule(backend="ffcv").
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        mode: str = "fine",
        target_type: Sequence[str] | str = _BaseCityscapes._DEFAULT_TARGET_TYPE,
        transforms=None,
        target_size: Tuple[int, int] | None = None,
        ffcv_path: str | Path | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            root=root,
            split=split,
            mode=mode,
            target_type=target_type,
            transforms=transforms,
            target_size=target_size,
            **kwargs,
        )
        self.ffcv_path = ffcv_path
