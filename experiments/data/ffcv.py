"""FFCV-specific utilities and helpers for segmentation datasets.

This module provides FFCV (Fast Forward Computer Vision) integration including:
- FFCV file format writing utilities
- FFCV-specific data loading wrappers
- Integration with the existing segmentation data module hierarchy
- Conversion utilities for common segmentation datasets
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import numpy as np
from ffcv.fields import BytesField, RGBImageField
from ffcv.writer import DatasetWriter

log = logging.getLogger(__name__)


class FFCVSegmentationWriter:
    """Writer for converting segmentation datasets to FFCV format.

    This class handles the conversion of (image, mask) pairs to FFCV's
    optimized file format for accelerated loading during training.
    """

    def __init__(
        self,
        output_path: str | Path,
        image_field: str = "image",
        label_field: str = "label",
        max_image_size: int | None = None,
    ):
        """Initialize the FFCV segmentation writer.

        Args:
            output_path: Path to the output .ffcv file.
            image_field: Field name for images in FFCV format.
            label_field: Field name for masks/labels in FFCV format.
            max_image_size: Maximum image dimension. Images will be resized
                to fit within this bound while preserving aspect ratio.
        """
        self.output_path = Path(output_path)
        self.image_field = image_field
        self.label_field = label_field
        self.max_image_size = max_image_size

    def write(self, dataset: Any, chunksize: int = 64) -> None:
        """Write a segmentation dataset to FFCV format.

        Args:
            dataset: Segmentation dataset that returns (image, mask) tuples.
                Images can be PIL Images, numpy arrays, or torch tensors.
            chunksize: Chunk size for processing during conversion.
        """
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        log.info(
            "Writing FFCV segmentation dataset to %s (%d samples)",
            self.output_path,
            len(dataset),
        )

        writer = DatasetWriter(
            str(self.output_path),
            {
                self.image_field: RGBImageField(),
                self.label_field: BytesField(),
            },
        )
        writer.from_indexed_dataset(dataset, chunksize=chunksize)

        log.info("FFCV file written successfully: %s", self.output_path)


def convert_segmentation_dataset_to_ffcv(
    dataset: Any,
    output_path: str | Path,
    image_field: str = "image",
    label_field: str = "label",
    chunksize: int = 64,
) -> Path:
    """Convert a segmentation dataset to FFCV format.

    Args:
        dataset: Segmentation dataset that returns (image, mask) tuples.
        output_path: Path to the output .ffcv file.
        image_field: Field name for images in FFCV format.
        label_field: Field name for masks/labels in FFCV format.
        chunksize: Chunk size for processing during conversion.

    Returns:
        Path to the created FFCV file.
    """
    writer = FFCVSegmentationWriter(
        output_path, image_field=image_field, label_field=label_field
    )
    writer.write(dataset, chunksize=chunksize)
    return Path(output_path)


def create_ffcv_dataloader(
    ffcv_path: str | Path,
    *,
    batch_size: int,
    num_workers: int = 0,
    distributed: bool = False,
    train: bool = True,
    image_pipeline: Optional[Callable] = None,
    label_pipeline: Optional[Callable] = None,
    in_memory: bool = False,
    seed: int = 42,
    **loader_kwargs,
) -> Any:
    """Create an FFCV DataLoader for segmentation datasets.

    Args:
        ffcv_path: Path to the .ffcv file.
        batch_size: Batch size.
        num_workers: Number of data loading workers.
        distributed: Whether to use distributed sampling.
        train: If True, use random sampling; otherwise sequential.
        image_pipeline: FFCV pipeline for image transformations.
        label_pipeline: FFCV pipeline for label transformations.
        in_memory: Whether to cache data in OS page cache.
        seed: Random seed for shuffling.
        **loader_kwargs: Additional arguments passed to FFCV Loader.

    Returns:
        FFCV Loader instance.
    """
    from ffcv.loader import Loader, OrderOption

    ffcv_path = Path(ffcv_path)
    if not ffcv_path.exists():
        raise FileNotFoundError(f"FFCV file not found: {ffcv_path}")

    order = OrderOption.QUASI_RANDOM if train else OrderOption.SEQUENTIAL

    pipelines = {}
    if image_pipeline is not None:
        pipelines["image"] = image_pipeline
    if label_pipeline is not None:
        pipelines["label"] = label_pipeline

    loader = Loader(
        str(ffcv_path),
        batch_size=batch_size,
        num_workers=num_workers,
        order=order,
        os_cache=in_memory,
        drop_last=train,
        pipelines=pipelines if pipelines else None,
        seed=seed,
        distributed=distributed,
        **loader_kwargs,
    )
    return loader


# ---------------------------------------------------------------------------
# Dataset-specific conversion utilities
# ---------------------------------------------------------------------------


class SegmentationDatasetWrapper:
    """Wrapper that adapts segmentation datasets for FFCV writing.

    Ensures consistent (image, mask) output format regardless of the
    underlying dataset backend (PIL, OpenCV, etc.).
    """

    def __init__(self, dataset: Any, target_size: tuple[int, int] | None = None):
        """Initialize the wrapper.

        Args:
            dataset: Segmentation dataset.
            target_size: Optional target size for resizing (width, height).
        """
        self.dataset = dataset
        self.target_size = target_size

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        image, mask = self.dataset[index]

        # Convert PIL Image to numpy array
        if hasattr(image, "convert"):
            # PIL Image
            image = np.array(image.convert("RGB"))
        elif isinstance(image, np.ndarray):
            # OpenCV image (BGR) - already numpy array
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 1:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert mask to numpy array
        if hasattr(mask, "convert"):
            # PIL Image
            mask = np.array(mask)
        elif isinstance(mask, np.ndarray):
            if mask.ndim == 3 and mask.shape[2] == 1:
                mask = mask.squeeze()
            elif mask.ndim == 3 and mask.shape[0] == 1:
                mask = mask.squeeze(0)

        # Resize if target_size specified
        if self.target_size is not None:
            image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)

        # Flatten mask to 1D uint8 array for BytesField
        mask_flat = mask.flatten().astype(np.uint8)

        return image, mask_flat


def convert_cityscapes_to_ffcv(
    root: str | Path,
    split: str = "train",
    output_path: str | Path | None = None,
    target_size: tuple[int, int] | None = None,
    mode: str = "fine",
) -> Path:
    """Convert Cityscapes dataset to FFCV format.

    Args:
        root: Root directory of Cityscapes dataset.
        split: Dataset split ("train", "val", "test").
        output_path: Output path for .ffcv file. If None, uses default naming.
        target_size: Optional target size for resizing (width, height).
        mode: Annotation mode ("fine" or "coarse").

    Returns:
        Path to the created FFCV file.
    """
    from .cityscapes import CV2Cityscapes

    root = Path(root)

    # Create dataset wrapper
    dataset = CV2Cityscapes(
        root=root,
        split=split,
        mode=mode,
        target_type="semantic_trainids",
        transforms=None,
        target_size=target_size,
    )

    wrapper = SegmentationDatasetWrapper(dataset, target_size=target_size)

    # Determine output path
    if output_path is None:
        output_path = root / f"cityscapes_{split}_{mode}.ffcv"

    return convert_segmentation_dataset_to_ffcv(wrapper, output_path, chunksize=64)


def convert_ade20k_to_ffcv(
    root: str | Path,
    split: str = "train",
    output_path: str | Path | None = None,
    target_size: tuple[int, int] | None = None,
) -> Path:
    """Convert ADE20K dataset to FFCV format.

    Args:
        root: Root directory of ADE20K dataset.
        split: Dataset split ("train" or "val").
        output_path: Output path for .ffcv file. If None, uses default naming.
        target_size: Optional target size for resizing (width, height).

    Returns:
        Path to the created FFCV file.
    """
    from .ade20k import CV2Ade20kSegmentation

    root = Path(root)

    # Create dataset wrapper
    dataset = CV2Ade20kSegmentation(
        root=root,
        split=split,
        transforms=None,
        target_size=target_size,
    )

    wrapper = SegmentationDatasetWrapper(dataset, target_size=target_size)

    # Determine output path
    if output_path is None:
        output_path = root / f"ade20k_{split}.ffcv"

    return convert_segmentation_dataset_to_ffcv(wrapper, output_path, chunksize=64)


def convert_voc_to_ffcv(
    root: str | Path,
    split: str = "train",
    year: str = "2012",
    output_path: str | Path | None = None,
    target_size: tuple[int, int] | None = None,
) -> Path:
    """Convert Pascal VOC dataset to FFCV format.

    Args:
        root: Root directory of Pascal VOC dataset.
        split: Dataset split ("train", "trainval", "val", "test").
        year: Dataset year ("2007", "2012", or "2007-test").
        output_path: Output path for .ffcv file. If None, uses default naming.
        target_size: Optional target size for resizing (width, height).

    Returns:
        Path to the created FFCV file.
    """
    from .pascal_voc import CV2VOCSegmentation

    root = Path(root)

    # Create dataset wrapper
    dataset = CV2VOCSegmentation(
        root=root,
        split=split,
        year=year,
        transforms=None,
        target_size=target_size,
    )

    wrapper = SegmentationDatasetWrapper(dataset, target_size=target_size)

    # Determine output path
    if output_path is None:
        output_path = root / f"voc_{year}_{split}.ffcv"

    return convert_segmentation_dataset_to_ffcv(wrapper, output_path, chunksize=64)


def convert_voc_aug_to_ffcv(
    root: str | Path,
    split: str = "train",
    year: str = "2012",
    output_path: str | Path | None = None,
    target_size: tuple[int, int] | None = None,
) -> Path:
    """Convert Pascal VOC Augmented dataset to FFCV format.

    Args:
        root: Root directory of Pascal VOC dataset.
        split: Dataset split ("train", "trainval", "val").
        year: Dataset year ("2012").
        output_path: Output path for .ffcv file. If None, uses default naming.
        target_size: Optional target size for resizing (width, height).

    Returns:
        Path to the created FFCV file.
    """
    from .pascal_aug import CV2VOCAugSegmentation

    root = Path(root)

    # Create dataset wrapper
    dataset = CV2VOCAugSegmentation(
        root=root,
        split=split,
        year=year,
        transforms=None,
    )

    wrapper = SegmentationDatasetWrapper(dataset, target_size=target_size)

    # Determine output path
    if output_path is None:
        output_path = root / f"voc_aug_{year}_{split}.ffcv"

    return convert_segmentation_dataset_to_ffcv(wrapper, output_path, chunksize=64)
