"""Base data modules for segmentation datasets with flexible backend support.

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

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import hydra
import rootutils
import torch
from lightning import LightningDataModule
from omegaconf import MISSING, DictConfig
from torch.utils.data import DataLoader, Dataset, Subset

log = logging.getLogger(__name__)


class DatasetSubset(Subset):
    """Subset wrapper that delegates attribute access to the underlying dataset.

    This is necessary because torch.utils.data.Subset only implements __getitem__
    and __len__, but our datasets have additional methods like sort(), get_class_names(),
    get_palette(), etc. that need to be accessible.
    """

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the underlying dataset."""
        if name in ("dataset", "indices"):
            # These are Subset's own attributes
            return object.__getattribute__(self, name)
        # Delegate to the underlying dataset
        return getattr(self.dataset, name)


def default_data_root() -> Path:
    """Return ``<project_root>/data`` using the ``.project-root`` marker."""
    root = rootutils.find_root(search_from=__file__, indicator=".project-root")
    return root / "data"


@dataclass
class DatasetInfo:
    """Base config for dataset information."""

    input_size: tuple[int, int] = MISSING
    input_min: float | tuple[float, float, float] = MISSING
    input_max: float | tuple[float, float, float] = MISSING
    num_classes: int = MISSING
    ignore_index: int = MISSING
    train_val_test_split: tuple[int, int, int | None] = MISSING
    mean: tuple[float, float, float] = MISSING
    std: tuple[float, float, float] = MISSING


def resolve_ffcv_file(
    root: str, explicit_path: str | None, candidates: tuple[str, ...]
) -> Path:
    """Resolve an FFCV .ffcv/.beton file from an explicit path or candidate filenames."""
    if explicit_path:
        path = Path(explicit_path)
        if path.is_file():
            return path
        raise FileNotFoundError(f"Configured FFCV file not found: {path}")

    root_path = Path(root)
    if root_path.is_file():
        return root_path

    for candidate in candidates:
        candidate_path = root_path / candidate
        if candidate_path.is_file():
            return candidate_path
    raise FileNotFoundError(
        f"Unable to resolve FFCV file under {root_path}. Tried: {', '.join(candidates)}"
    )


def ffcv_file_exists(
    root: str, explicit_path: str | None, candidates: tuple[str, ...]
) -> bool:
    """Return True if an FFCV file can be resolved, False otherwise."""
    try:
        resolve_ffcv_file(root, explicit_path, candidates)
        return True
    except FileNotFoundError:
        return False


def write_ffcv_segmentation_dataset(
    dataset: Any,
    path: str | Path,
    image_field: str = "image",
    label_field: str = "label",
) -> None:
    """Write a segmentation dataset to FFCV format at *path*.

    Args:
        dataset: Segmentation dataset with (image, mask) pairs.
            Images should be RGB numpy arrays (H, W, 3).
            Masks should be 2D numpy arrays (H, W) with integer class labels.
        path: Output path for the FFCV file.
        image_field: Field name for images in FFCV format.
        label_field: Field name for masks/labels in FFCV format.
    """
    import numpy as np
    from ffcv.fields import BytesField, RGBImageField
    from ffcv.writer import DatasetWriter

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Writing FFCV segmentation dataset to %s (%d samples)", path, len(dataset))

    # Create a wrapper that ensures proper format for FFCV
    class FFCVWrapper:
        def __init__(self, dataset):
            self.dataset = dataset

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            image, mask = self.dataset[idx]
            # Ensure image is uint8 RGB
            if hasattr(image, "numpy"):
                image = image.numpy()
            image = np.asarray(image, dtype=np.uint8)
            if image.ndim == 2:
                image = np.stack([image] * 3, axis=-1)

            # Flatten mask to 1D uint8 array for BytesField
            if hasattr(mask, "numpy"):
                mask = mask.numpy()
            mask = np.asarray(mask, dtype=np.uint8)
            if mask.ndim == 3 and mask.shape[-1] == 1:
                mask = mask.squeeze(-1)
            mask_flat = mask.flatten()

            return image, mask_flat

    wrapped_dataset = FFCVWrapper(dataset)

    writer = DatasetWriter(
        str(path),
        {
            image_field: RGBImageField(),
            label_field: BytesField(),
        },
    )
    writer.from_indexed_dataset(wrapped_dataset)


def _instantiate_pipeline(pipeline_cfg: Any) -> Any:
    """Instantiate a pipeline from a Hydra config or return an already-built object."""
    if pipeline_cfg is None:
        return None
    if isinstance(pipeline_cfg, (DictConfig, dict)):
        return hydra.utils.instantiate(pipeline_cfg)
    return pipeline_cfg


# ---------------------------------------------------------------------------
# BaseSegmentationDataModule — abstract base for segmentation data modules
# ---------------------------------------------------------------------------


class BaseSegmentationDataModule(LightningDataModule):
    """Abstract base data module for segmentation datasets.

    Provides the shared interface for dataset construction, validation
    splitting, and metadata. Subclasses implement the backend-specific
    ``prepare_data``, ``setup``, and dataloader methods.

    Parameters
    ----------
    val_split_mode : str
        How to split training data when ``val_size > 0``:
        * ``"random"`` — Random permutation (seeded).
        * ``"sequential"`` — Last *val_size* samples become validation
          (e.g., Cityscapes 2975 → 2475 train + 500 val).
    max_train_samples : int | None
        Maximum number of training samples to use. If None, uses all available
        samples. Subset selection is reproducible based on the seed parameter.
    max_val_samples : int | None
        Maximum number of validation samples to use. If None, uses all available
        samples. Subset selection is reproducible based on the seed parameter.
    max_test_samples : int | None
        Maximum number of test samples to use. If None, uses all available
        samples. Subset selection is reproducible based on the seed parameter.
    seed : int
        Random seed for reproducible subset selection and shuffling (default: 42).
    """

    # Override in subclasses
    info: DatasetInfo | None = None

    def __init__(
        self,
        *,
        dataset: DictConfig | None = None,
        root: str | None = None,
        train_pipeline: Any = None,
        val_pipeline: Any = None,
        test_pipeline: Any = None,
        batch_transform: Any = None,
        train_split: str = "train",
        val_split: str = "val",
        test_split: str | None = None,
        val_size: int = 0,
        val_split_mode: str = "random",
        batch_size: int = 4,
        test_batch_size: int = 4,
        num_workers: int = 0,
        pin_memory: bool = False,
        seed: int = 42,
        name: str = "",
        max_train_samples: int | None = None,
        max_val_samples: int | None = None,
        max_test_samples: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        # Dataset configuration for Hydra instantiation
        if dataset is not None:
            dataset = dict(dataset)  # Mutable copy
            self._dataset_class: str | None = dataset.pop("class_path")
            self._dataset_cfg: dict[str, Any] = dataset
        else:
            self._dataset_class = None
            self._dataset_cfg = {}

        self._root = root or self._dataset_cfg.get("root", "")

        # Batch-level transform (e.g., cutmix/mixup for segmentation), instantiated lazily
        self._batch_transform_cfg = batch_transform
        self._batch_transform = None

        self._pipelines = {
            "train": train_pipeline,
            "val": val_pipeline,
            "test": test_pipeline if test_pipeline is not None else val_pipeline,
        }
        self._splits = {
            "train": train_split,
            "val": val_split,
            "test": test_split if test_split is not None else val_split,
        }

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.g = torch.Generator().manual_seed(seed)
        self.save_hyperparameters(logger=False)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def root(self) -> str:
        return self._root

    @property
    def num_classes(self) -> int | None:
        if self.info is not None:
            return self.info.num_classes
        return None

    @property
    def dataset_name(self) -> str:
        return self.hparams.get("name", "")

    # ------------------------------------------------------------------
    # Dataset construction hooks (override in subclasses / mixins)
    # ------------------------------------------------------------------

    def _make_raw_dataset(self, split: str) -> Any:
        """Create a raw dataset without transforms.

        Used for downloading data and for writing FFCV files.
        If a ``dataset`` config was provided, uses Hydra instantiation.
        Otherwise, subclasses must override this.
        """
        if self._dataset_class is None:
            raise NotImplementedError(
                f"{type(self).__name__} must override _make_raw_dataset() or "
                "provide a dataset config with class_path."
            )
        return hydra.utils.instantiate(
            {"_target_": self._dataset_class, **self._dataset_cfg},
            split=split,
            transforms=None,
            _convert_="partial",
        )

    def _make_dataset(self, split: str, transforms: Any) -> Dataset:
        """Create a dataset with transforms applied.

        Override in subclasses that don't use Hydra instantiation.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must override _make_dataset() or "
            "provide a dataset config with class_path."
        )

    def _build_dataset(
        self, stage: str, override_pipeline: str | None = None
    ) -> Dataset:
        """Build a dataset for the given stage with the configured pipeline."""
        split = self._splits.get(stage)
        pipeline_key = override_pipeline or stage
        transforms = _instantiate_pipeline(self._pipelines.get(pipeline_key))

        if self._dataset_class is not None:
            dataset = hydra.utils.instantiate(
                {"_target_": self._dataset_class, **self._dataset_cfg},
                split=split,
                transforms=transforms,
                _convert_="partial",
            )
        else:
            dataset = self._make_dataset(split, transforms)

        dataset_info = getattr(dataset, "info", None)
        if dataset_info is not None and self.info is None:
            self.info = dataset_info
        return dataset

    # ------------------------------------------------------------------
    # Batch-level transform (for segmentation-specific augmentations)
    # ------------------------------------------------------------------

    def on_after_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        if self.training and self._batch_transform_cfg is not None:
            if self._batch_transform is None:
                self._batch_transform = _instantiate_pipeline(self._batch_transform_cfg)
            images, labels = batch
            images, labels = self._batch_transform(images, labels)
            return images, labels
        return batch

    # ------------------------------------------------------------------
    # Validation split helper
    # ------------------------------------------------------------------

    def _split_train_val(
        self,
        train_dataset: Dataset,
        eval_dataset: Dataset,
    ) -> tuple[Subset, Subset]:
        """Split training data into train/val subsets.

        Parameters
        ----------
        train_dataset : Dataset
            Full training set (with train augmentations).
        eval_dataset : Dataset
            Same underlying data but with eval augmentations.

        Returns
        -------
        (train_subset, val_subset)
        """
        total = len(train_dataset)
        val_size = self.hparams.val_size

        if self.hparams.val_split_mode == "sequential":
            train_indices = list(range(total - val_size))
            val_indices = list(range(total - val_size, total))
        else:
            generator = torch.Generator().manual_seed(self.hparams.seed)
            perm = torch.randperm(total, generator=generator).tolist()
            val_indices = perm[:val_size]
            train_indices = perm[val_size:]

        return DatasetSubset(train_dataset, train_indices), DatasetSubset(
            eval_dataset, val_indices
        )

    def _create_subset(
        self,
        dataset: Dataset,
        max_samples: int | None,
        seed: int | None = None,
    ) -> Dataset:
        """Create a reproducible subset of a dataset.

        Parameters
        ----------
        dataset : Dataset
            Full dataset to sample from.
        max_samples : int | None
            Maximum number of samples to include. If None or >= len(dataset),
            returns the full dataset.
        seed : int | None
            Random seed for reproducibility. If None, uses self.hparams.seed.

        Returns
        -------
        Dataset
            Subset of the dataset with at most max_samples elements, or the
            original dataset if max_samples is None or >= len(dataset).
        """
        if max_samples is None or max_samples >= len(dataset):
            return dataset

        total = len(dataset)
        if seed is None:
            seed = self.hparams.seed

        # Create reproducible random indices
        generator = torch.Generator().manual_seed(seed)
        perm = torch.randperm(total, generator=generator).tolist()
        indices = perm[:max_samples]

        return DatasetSubset(dataset, indices)

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def get_weights(self) -> list[float]:
        """Get class weights from the training dataset."""
        if self.data_train is None:
            dataset = self._build_dataset("train")
        else:
            dataset = self.data_train

        if hasattr(dataset, "get_weights"):
            weights = dataset.get_weights()
            if weights is None:
                raise ValueError("Dataset did not return any class weights.")
            return weights
        raise ValueError("Dataset does not define class weights.")


# ---------------------------------------------------------------------------
# SegmentationDataModule — data module for OpenCV and PIL backends
# ---------------------------------------------------------------------------


class SegmentationDataModule(BaseSegmentationDataModule):
    """Unified data module for segmentation datasets with OpenCV or PIL support.

    Parameters
    ----------
    backend : str
        Backend to use: "opencv", "pil". Defaults to "opencv".
    """

    def __init__(
        self,
        *,
        backend: str = "opencv",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.backend = backend.lower()

    @property
    def backend_name(self) -> str:
        return self.backend

    def prepare_data(self) -> None:
        for split in {s for s in self._splits.values() if s is not None}:
            self._make_raw_dataset(split)

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit") and self.data_train is None:
            self.data_train = self._build_dataset("train")

            if self.hparams.val_size > 0:
                eval_dataset = self._build_dataset("train", override_pipeline="val")
                self.data_train, self.data_val = self._split_train_val(
                    self.data_train, eval_dataset
                )
            else:
                self.data_val = self._build_dataset("val")

            # Apply subset selection
            self.data_train = self._create_subset(
                self.data_train, self.hparams.max_train_samples
            )
            self.data_val = self._create_subset(
                self.data_val, self.hparams.max_val_samples
            )

        if stage in (None, "test") and self.data_test is None:
            self.data_test = self._build_dataset("test")
            # Apply subset selection
            self.data_test = self._create_subset(
                self.data_test, self.hparams.max_test_samples
            )

    def _make_dataloader(
        self,
        dataset: Optional[Dataset],
        batch_size: int,
        shuffle: bool = False,
        drop_last: bool = False,
    ) -> DataLoader:
        if dataset is None:
            raise RuntimeError("Dataset has not been set up yet.")

        generator = (
            torch.Generator().manual_seed(self.hparams.seed) if shuffle else None
        )

        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            drop_last=drop_last,
            generator=generator,
        )

    def train_dataloader(self, shuffle: bool = True) -> DataLoader:
        return self._make_dataloader(
            self.data_train,
            self.hparams.batch_size,
            shuffle=shuffle,
            drop_last=True,
        )

    def val_dataloader(self, shuffle: bool = False) -> DataLoader:
        if self.data_val is None:
            raise RuntimeError("No validation set configured.")
        return self._make_dataloader(
            self.data_val, self.hparams.test_batch_size, shuffle=shuffle
        )

    def test_dataloader(self, shuffle: bool = False) -> DataLoader:
        return self._make_dataloader(
            self.data_test, self.hparams.test_batch_size, shuffle=shuffle
        )


# ---------------------------------------------------------------------------
# DALISegmentationDataModule — NVIDIA DALI data module
# ---------------------------------------------------------------------------


class DALISegmentationDataModule(BaseSegmentationDataModule):
    """NVIDIA DALI data module for GPU-accelerated data loading.

    DALI provides GPU-accelerated image decoding, augmentation, and preprocessing
    with support for multiple backends and efficient pipelining.
    """

    def __init__(
        self,
        *,
        device: str = "gpu",
        device_id: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.backend = "dali"
        self.device = device
        self.device_id = device_id

        self._train_iterator: Any = None
        self._val_iterator: Any = None
        self._test_iterator: Any = None

    @property
    def backend_name(self) -> str:
        return self.backend

    def _get_file_paths(
        self, split: str, max_samples: int | None = None
    ) -> tuple[list[str], list[str]]:
        """Get image and mask file paths for a split.

        Parameters
        ----------
        split : str
            Dataset split name (train/val/test).
        max_samples : int | None
            Maximum number of samples to include. If None, returns all samples.

        Returns
        -------
        tuple[list[str], list[str]]
            Lists of image paths and mask paths.
        """
        dataset = self._make_raw_dataset(split)
        images = getattr(dataset, "images", [])
        masks = getattr(dataset, "masks", [])

        # Apply subset selection if requested
        if max_samples is not None and max_samples < len(images):
            # Create reproducible random indices
            generator = torch.Generator().manual_seed(self.hparams.seed)
            total = len(images)
            perm = torch.randperm(total, generator=generator).tolist()
            indices = perm[:max_samples]

            images = [images[i] for i in indices]
            masks = [masks[i] for i in indices]

        return [str(p) for p in images], [str(p) for p in masks]

    def setup(self, stage: Optional[str] = None) -> None:
        from data.pipelines import (
            DALISegmentationEvalPipeline,
            DALISegmentationTrainPipeline,
        )

        if stage in (None, "fit") and self._train_iterator is None:
            # Get file paths with subset selection
            train_images, train_masks = self._get_file_paths(
                "train", max_samples=self.hparams.max_train_samples
            )

            # Build train pipeline
            train_pipeline = _instantiate_pipeline(self._pipelines["train"])
            if train_pipeline is None:
                train_pipeline = DALISegmentationTrainPipeline(
                    base_size=self.hparams.base_size,
                    crop_size=self.hparams.crop_size,
                    random_scale=True,
                    scale_min=0.5,
                    scale_max=2.0,
                    random_flip=True,
                    random_crop=True,
                    normalize=self.hparams.normalize,
                    to_tensor=self.hparams.to_tensor,
                    device=self.device,
                    device_id=self.device_id,
                )

            self._train_iterator = train_pipeline.create_iterator(
                train_images,
                train_masks,
                batch_size=self.hparams.batch_size,
                shuffle=True,
            )

            # Build val pipeline with subset selection
            val_images, val_masks = self._get_file_paths(
                "val", max_samples=self.hparams.max_val_samples
            )
            val_pipeline = _instantiate_pipeline(self._pipelines["val"])
            if val_pipeline is None:
                val_pipeline = DALISegmentationEvalPipeline(
                    base_size=self.hparams.base_size,
                    crop_size=self.hparams.crop_size,
                    center_crop=False,
                    normalize=self.hparams.normalize,
                    to_tensor=self.hparams.to_tensor,
                    device=self.device,
                    device_id=self.device_id,
                )

            self._val_iterator = val_pipeline.create_iterator(
                val_images,
                val_masks,
                batch_size=self.hparams.test_batch_size,
            )

        if stage in (None, "test") and self._test_iterator is None:
            # Get test file paths with subset selection
            test_images, test_masks = self._get_file_paths(
                "test", max_samples=self.hparams.max_test_samples
            )
            test_pipeline = _instantiate_pipeline(self._pipelines["test"])
            if test_pipeline is None:
                test_pipeline = DALISegmentationEvalPipeline(
                    base_size=self.hparams.base_size,
                    crop_size=self.hparams.crop_size,
                    center_crop=False,
                    normalize=self.hparams.normalize,
                    to_tensor=self.hparams.to_tensor,
                    device=self.device,
                    device_id=self.device_id,
                )

            self._test_iterator = test_pipeline.create_iterator(
                test_images,
                test_masks,
                batch_size=self.hparams.test_batch_size,
            )

    def train_dataloader(self, shuffle: bool = True) -> Any:
        if self._train_iterator is None:
            raise RuntimeError(
                "DALI training iterator not initialized. Call setup() first."
            )
        return self._train_iterator

    def val_dataloader(self, shuffle: bool = False) -> Any:
        if self._val_iterator is None:
            raise RuntimeError(
                "DALI validation iterator not initialized. Call setup() first."
            )
        return self._val_iterator

    def test_dataloader(self, shuffle: bool = False) -> Any:
        if self._test_iterator is None:
            raise RuntimeError(
                "DALI test iterator not initialized. Call setup() first."
            )
        return self._test_iterator


# ---------------------------------------------------------------------------
# FFCVSegmentationDataModule — FFCV specific data module
# ---------------------------------------------------------------------------


class _FFCVLabelReshapeWrapper:
    """Wrapper that reshapes flattened masks from FFCV BytesDecoder.

    FFCV's BytesDecoder returns flattened 1D arrays. This wrapper reshapes
    them back to 2D (H, W) based on the corresponding image dimensions.
    """

    def __init__(self, loader: Any):
        self.loader = loader

    def __len__(self) -> int:
        return len(self.loader)

    def __iter__(self):
        for batch in self.loader:
            # FFCV returns (images, labels) where labels are flattened
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                images, labels = batch
                # Reshape labels to match image spatial dimensions
                if isinstance(images, torch.Tensor) and isinstance(
                    labels, torch.Tensor
                ):
                    # images: (B, C, H, W), labels: (B, H*W) or (B, H, W)
                    if labels.ndim == 2 and images.ndim == 4:
                        h, w = images.shape[2], images.shape[3]
                        labels = labels.view(-1, h, w)
                yield images, labels
            else:
                yield batch


class FFCVSegmentationDataModule(BaseSegmentationDataModule):
    """FFCV data module for accelerated GPU-direct loading from .ffcv files.

    Uses FFCV's native GPU-accelerated augmentations for maximum performance.
    """

    def __init__(
        self,
        *,
        train_dataset: str | None = None,
        val_dataset: str | None = None,
        train_ffcv_filenames: tuple[str, ...] = ("train.ffcv", "train.beton"),
        eval_ffcv_filenames: tuple[str, ...] = ("test.ffcv", "val.ffcv", "val.beton"),
        in_memory: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.backend = "ffcv"
        self._train_dataset_path = train_dataset
        self._val_dataset_path = val_dataset
        self.train_ffcv_filenames = train_ffcv_filenames
        self.eval_ffcv_filenames = eval_ffcv_filenames
        self.in_memory = in_memory

        self._train_loader: Any = None
        self._val_loader: Any = None
        self._test_loader: Any = None

    @property
    def backend_name(self) -> str:
        return self.backend

    def prepare_data(self) -> None:
        for split in {s for s in self._splits.values() if s is not None}:
            if split == self._splits.get("train"):
                candidates = self.train_ffcv_filenames
                output_path = self._train_dataset_path
            else:
                candidates = self.eval_ffcv_filenames
                output_path = self._val_dataset_path

            if ffcv_file_exists(self._root, output_path, candidates):
                log.info("FFCV file already exists for split: %s", split)
                continue

            log.info("Creating FFCV file for split: %s", split)
            dataset = self._make_raw_dataset(split)

            output_file = output_path or str(Path(self._root) / candidates[0])
            write_ffcv_segmentation_dataset(dataset, output_file)

    def _create_ffcv_loader(
        self,
        explicit_path: str | None,
        candidates: tuple[str, ...],
        batch_size: int,
        shuffle: bool,
        pipeline: Any | None,
        order: str | None = None,
        drop_last: bool = False,
        max_samples: int | None = None,
    ) -> Any:
        from ffcv.loader import Loader, OrderOption

        ffcv_path = resolve_ffcv_file(self._root, explicit_path, candidates)

        if order is not None:
            order_option = (
                OrderOption.QUASI_RANDOM
                if order == "quasi_random"
                else OrderOption.SEQUENTIAL
            )
        elif shuffle:
            order_option = OrderOption.QUASI_RANDOM
        else:
            order_option = OrderOption.SEQUENTIAL

        pipelines = {}
        if pipeline is not None:
            if hasattr(pipeline, "build_image_pipeline"):
                img_pipe = pipeline.build_image_pipeline()
                if img_pipe is not None:
                    pipelines["image"] = img_pipe
            if hasattr(pipeline, "build_label_pipeline"):
                label_pipe = pipeline.build_label_pipeline()
                if label_pipe is not None:
                    pipelines["label"] = label_pipe

        # Create subset indices if max_samples is specified
        indices = None
        if max_samples is not None:
            # We need to get the dataset size first to create indices
            # FFCV doesn't expose dataset length before creating a loader,
            # so we create a temporary loader to get the size
            temp_loader = Loader(
                str(ffcv_path),
                batch_size=1,
                num_workers=0,
                order=OrderOption.SEQUENTIAL,
                pipelines=None,
                distributed=False,
            )
            total_samples = len(temp_loader)

            if max_samples < total_samples:
                # Create reproducible random indices
                generator = torch.Generator().manual_seed(self.hparams.seed)
                perm = torch.randperm(total_samples, generator=generator).tolist()
                indices = perm[:max_samples]

        loader = Loader(
            str(ffcv_path),
            batch_size=batch_size,
            num_workers=self.hparams.num_workers,
            order=order_option,
            os_cache=self.in_memory,
            drop_last=drop_last,
            pipelines=pipelines if pipelines else None,
            seed=self.hparams.seed,
            distributed=False,
            indices=indices,
        )

        # Wrap loader to reshape flattened masks from BytesDecoder
        return _FFCVLabelReshapeWrapper(loader)

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit") and self._train_loader is None:
            train_pipeline = _instantiate_pipeline(self._pipelines["train"])
            self._train_loader = self._create_ffcv_loader(
                explicit_path=self._train_dataset_path,
                candidates=self.train_ffcv_filenames,
                batch_size=self.hparams.batch_size,
                shuffle=True,
                pipeline=train_pipeline,
                order="quasi_random",
                drop_last=True,
                max_samples=self.hparams.max_train_samples,
            )

            val_pipeline = _instantiate_pipeline(self._pipelines["val"])
            self._val_loader = self._create_ffcv_loader(
                explicit_path=self._val_dataset_path,
                candidates=self.eval_ffcv_filenames,
                batch_size=self.hparams.test_batch_size,
                shuffle=False,
                pipeline=val_pipeline,
                order="sequential",
                drop_last=False,
                max_samples=self.hparams.max_val_samples,
            )

        if stage in (None, "test") and self._test_loader is None:
            test_pipeline = _instantiate_pipeline(self._pipelines["test"])
            self._test_loader = self._create_ffcv_loader(
                explicit_path=self._val_dataset_path,
                candidates=self.eval_ffcv_filenames,
                batch_size=self.hparams.test_batch_size,
                shuffle=False,
                pipeline=test_pipeline,
                order="sequential",
                drop_last=False,
                max_samples=self.hparams.max_test_samples,
            )

    def train_dataloader(self, shuffle: bool = True) -> Any:
        if self._train_loader is None:
            raise RuntimeError(
                "FFCV training loader not initialized. Call setup() first."
            )
        return self._train_loader

    def val_dataloader(self, shuffle: bool = False) -> Any:
        if self._val_loader is None:
            raise RuntimeError(
                "FFCV validation loader not initialized. Call setup() first."
            )
        return self._val_loader

    def test_dataloader(self, shuffle: bool = False) -> Any:
        if self._test_loader is None:
            raise RuntimeError("FFCV test loader not initialized. Call setup() first.")
        return self._test_loader
