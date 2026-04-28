"""Reusable mixins for loading images for segmentation datasets."""

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image


class SegmentationLoaderMixin:
    """Provides helpers for retrieving image and mask paths.

    Dataset classes are expected to define ``self.images`` and ``self.masks`` (or
    override the helper methods).
    """

    images: Sequence[str | Path]
    masks: Sequence[str | Path]
    target_size: Optional[Tuple[int, int]] = None

    def _get_image_path(self, index: int) -> str:
        return str(self.images[index])

    def _get_mask_path(self, index: int) -> str:
        return str(self.masks[index])

    @abstractmethod
    def _load_input(self, index: int) -> tuple[Any, Any]:
        raise NotImplementedError


class CV2SegmentationLoaderMixin(SegmentationLoaderMixin):
    """Mixin that loads segmentation samples with OpenCV."""

    float_images: bool = False

    def _load_input(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        image_path = self._get_image_path(index)
        mask_path = self._get_mask_path(index)

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if getattr(self, "float_images", False):
            image = np.float32(image)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if self.target_size is not None:
            image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)

        return image, mask


class PILSegmentationLoaderMixin(SegmentationLoaderMixin):
    """Mixin that loads segmentation samples using PIL."""

    def _load_input(self, index: int):
        image_path = self._get_image_path(index)
        mask_path = self._get_mask_path(index)

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)

        if self.target_size is not None:
            image = image.resize(self.target_size, Image.BILINEAR)
            mask = mask.resize(self.target_size, Image.NEAREST)

        return image, mask


class ClassInfoMixin:
    """Mixin to expose class names, colors, and palettes for segmentation
    datasets."""

    classes: Sequence[Any]

    @classmethod
    def get_class_names(cls) -> list[str]:
        names: list[str] = []
        for k in cls.classes:
            name = getattr(k, "name", None)
            if name is None:
                raise ValueError("Class name is not defined for all classes.")
            names.append(str(name))
        return names

    @classmethod
    def get_class_colors(cls) -> list[tuple[int, int, int]]:
        colors: list[tuple[int, int, int]] = []
        for k in cls.classes:
            color = getattr(k, "color", None)
            if color is None:
                raise ValueError("Class color is not defined for all classes.")
            colors.append(tuple(int(c) for c in color))
        return colors

    @classmethod
    def get_label_offset(cls) -> int:
        """Return the smallest class id if available (e.g., ADE20K starts at
        1)."""
        ids = []
        for k in cls.classes:
            cid = getattr(k, "id", None)
            if cid is None:
                raise ValueError("Class id is not defined for all classes.")
            ids.append(int(cid))
        return min(ids)

    @classmethod
    def get_palette(cls, pad_to: int | None = None) -> np.ndarray:
        """Build a palette aligned with label indices, optionally padded to
        ``pad_to`` entries."""
        colors = np.array(cls.get_class_colors(), dtype=np.uint8)
        offset = cls.get_label_offset()
        if offset > 0:
            colors = np.vstack([np.zeros((offset, 3), dtype=np.uint8), colors])
        return colors
