"""Data module for segmentation datasets with flexible backend support.

This module provides a unified interface for loading segmentation datasets
with support for multiple backends:
- OpenCV (cv2): Fast image loading with NumPy arrays
- PIL (Pillow): Standard PIL image loading
- FFCV: Accelerated loading using FFCV format files

Example usage:
    # Using OpenCV backend (default)
    data = SegmentationDataModule(
        backend="opencv",
        dataset={"class_path": "physical_attacks.data.cityscapes.CV2Cityscapes"},
        root="./data/cityscapes",
        train_pipeline=CV2SegmentationTrainPipeline(...),
    )

    # Using PIL backend
    data = SegmentationDataModule(
        backend="pil",
        dataset={"class_path": "physical_attacks.data.cityscapes.PILCityscapes"},
        root="./data/cityscapes",
        train_pipeline=PILSegmentationTrainPipeline(...),
    )

    # Using FFCV backend
    data = FFCVSegmentationDataModule(
        dataset={"class_path": "physical_attacks.data.cityscapes.CV2Cityscapes"},
        root="./data/cityscapes",
        train_pipeline=FFCVSegmentationTrainPipeline(...),
        train_dataset="./data/cityscapes/train.ffcv",
        val_dataset="./data/cityscapes/val.ffcv",
    )
"""

from .ade20k import (
    CV2Ade20kSegmentation,
    FFCVAde20kSegmentation,
    PILAde20kSegmentation,
    _BaseAde20kSegmentation,
)
from .base import (
    BaseSegmentationDataModule,
    DatasetInfo,
    FFCVSegmentationDataModule,
    SegmentationDataModule,
    default_data_root,
    ffcv_file_exists,
    resolve_ffcv_file,
    write_ffcv_segmentation_dataset,
)
from .cityscapes import (
    CV2Cityscapes,
    FFCVCityscapes,
    PILCityscapes,
    _BaseCityscapes,
)
from .configs import (
    register_configs,
)
from .ffcv import (
    FFCVSegmentationWriter,
    SegmentationDatasetWrapper,
    convert_ade20k_to_ffcv,
    convert_cityscapes_to_ffcv,
    convert_segmentation_dataset_to_ffcv,
    convert_voc_aug_to_ffcv,
    convert_voc_to_ffcv,
    create_ffcv_dataloader,
)
from .mixins import (
    ClassInfoMixin,
    CV2SegmentationLoaderMixin,
    PILSegmentationLoaderMixin,
    SegmentationLoaderMixin,
)
from .pascal_aug import (
    CV2VOCAugSegmentation,
    FFCVVOCAugSegmentation,
    PILVOCAugSegmentation,
    _BaseVOCAugSegmentation,
)
from .pascal_voc import (
    CV2VOCSegmentation,
    FFCVVOCSegmentation,
    PILVOCSegmentation,
    _BaseVOCSegmentation,
)
from .pipelines import (
    CV2SegmentationEvalPipeline,
    CV2SegmentationTrainPipeline,
    FFCVSegmentationEvalPipeline,
    FFCVSegmentationTrainPipeline,
    PILSegmentationEvalPipeline,
    PILSegmentationTrainPipeline,
)
from .transforms import PhotoMetricDistortion

__all__ = [
    # Base classes
    "DatasetInfo",
    "BaseSegmentationDataModule",
    "SegmentationDataModule",
    "TorchvisionSegmentationDataModule",  # Alias for SegmentationDataModule
    "FFCVSegmentationDataModule",
    # Data root utilities
    "default_data_root",
    "resolve_ffcv_file",
    "ffcv_file_exists",
    "write_ffcv_segmentation_dataset",
    "convert_segmentation_dataset_to_ffcv",  # Mixins
    "SegmentationLoaderMixin",
    "CV2SegmentationLoaderMixin",
    "PILSegmentationLoaderMixin",
    "ClassInfoMixin",
    # Pipelines
    "CV2SegmentationTrainPipeline",
    "CV2SegmentationEvalPipeline",
    "PILSegmentationTrainPipeline",
    "PILSegmentationEvalPipeline",
    "FFCVSegmentationTrainPipeline",
    "FFCVSegmentationEvalPipeline",
    # Transforms
    "PhotoMetricDistortion",
    # FFCV utilities
    "FFCVSegmentationWriter",
    "SegmentationDatasetWrapper",
    "create_ffcv_dataloader",
    # Dataset conversion utilities
    "convert_cityscapes_to_ffcv",
    "convert_ade20k_to_ffcv",
    "convert_voc_to_ffcv",
    "convert_voc_aug_to_ffcv",
    # Hydra config store
    "register_configs",
    # Cityscapes
    "_BaseCityscapes",
    "CV2Cityscapes",
    "PILCityscapes",
    "FFCVCityscapes",
    # ADE20K
    "_BaseAde20kSegmentation",
    "CV2Ade20kSegmentation",
    "PILAde20kSegmentation",
    "FFCVAde20kSegmentation",
    # Pascal VOC
    "_BaseVOCSegmentation",
    "CV2VOCSegmentation",
    "PILVOCSegmentation",
    "FFCVVOCSegmentation",
    # Pascal VOC Aug
    "_BaseVOCAugSegmentation",
    "CV2VOCAugSegmentation",
    "PILVOCAugSegmentation",
    "FFCVVOCAugSegmentation",
]
