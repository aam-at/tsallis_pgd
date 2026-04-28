import os
from pathlib import Path
from typing import Union

from torchvision.datasets.voc import DATASET_YEAR_DICT

from .base import DatasetInfo
from .configs import DatasetInfo, VOCAugInfo
from .mixins import (
    CV2SegmentationLoaderMixin,
    PILSegmentationLoaderMixin,
)
from .pascal_voc import _BaseVOCSegmentation


class _BaseVOCAugSegmentation(_BaseVOCSegmentation):
    _TARGET_DIR = "SegmentationClassAug"
    info: DatasetInfo = VOCAugInfo()

    def __init__(
        self,
        root: Union[str, Path],
        split: str = "train",
        year: str = "2012",
        transforms=None,
        **kwargs,
    ):
        super().__init__(
            root=root, split=split, year=year, transforms=transforms, **kwargs
        )

        key = "2007-test" if self.year == "2007" and split == "test" else year
        dataset_year_dict = DATASET_YEAR_DICT[key]

        base_dir = dataset_year_dict["base_dir"]
        voc_root = os.path.join(self.root, base_dir)

        splits_dir = os.path.join(voc_root, "ImageSets", self._SPLITS_DIR)
        split_name = split
        if split_name in ["train", "trainval"]:
            split_name += "_aug"
        split_f = os.path.join(splits_dir, split_name.rstrip("\n") + ".txt")
        with open(os.path.join(split_f)) as f:
            file_names = [x.strip() for x in f.readlines()]

        image_dir = os.path.join(voc_root, "JPEGImages")
        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]

        target_dir = os.path.join(voc_root, self._TARGET_DIR)
        self.targets = [
            os.path.join(target_dir, x + self._TARGET_FILE_EXT) for x in file_names
        ]

        assert len(self.images) == len(self.targets)


class CV2VOCAugSegmentation(CV2SegmentationLoaderMixin, _BaseVOCAugSegmentation):
    def __init__(
        self,
        root: str | Path,
        transforms=None,
        float_images: bool = False,
        **kwargs,
    ):
        super().__init__(root, transforms=transforms, **kwargs)
        self.float_images = float_images


class PILVOCAugSegmentation(PILSegmentationLoaderMixin, _BaseVOCAugSegmentation): ...


class FFCVVOCAugSegmentation(CV2SegmentationLoaderMixin, _BaseVOCAugSegmentation):
    """Pascal VOC Augmented dataset for creating FFCV format files.

    This class is used during FFCV file creation (prepare_data phase).
    For actual training with FFCV, use FFCVSegmentationDataModule(...).
    """

    def __init__(
        self,
        root: str | Path,
        transforms=None,
        ffcv_path: str | Path | None = None,
        **kwargs,
    ):
        super().__init__(root, transforms=transforms, **kwargs)
        self.ffcv_path = ffcv_path
