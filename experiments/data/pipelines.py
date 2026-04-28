from __future__ import annotations

import random
from typing import Sequence

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from nvidia.dali import fn, types
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from PIL import Image, ImageFilter, ImageOps

from .transforms import RandomGaussianBlur


class CV2SegmentationTrainPipeline:
    """Reusable training pipeline composed of Albumentations transforms."""

    def __init__(
        self,
        *,
        base_size: int,
        crop_size: int,
        pad_value: Sequence[float] | float = 0.0,
        ignore_index: int = 255,
        random_scale: bool = False,
        scale_min: float = 0.5,
        scale_max: float = 2.0,
        random_rotate: bool = False,
        rotate_min: int = -10,
        rotate_max: int = 10,
        random_gaussian_blur: bool = False,
        random_flip: bool = False,
        flip_prob: float = 0.5,
        random_crop: bool = True,
        normalize: bool = False,
        to_tensor: bool = True,
        is_check_shapes: bool = True,
    ):
        transforms_list = []
        # ----- Random scale -----
        if random_scale:
            transforms_list.append(
                A.RandomScale(scale_limit=(scale_min - 1, scale_max - 1), p=1.0)
            )

        # ----- Pad if too small -----
        transforms_list.append(
            A.PadIfNeeded(
                min_height=crop_size,
                min_width=crop_size,
                border_mode=cv2.BORDER_CONSTANT,
                position="top_left",
                value=pad_value,
                mask_value=ignore_index,
            )
        )

        # ----- Random rotate -----
        if random_rotate:
            transforms_list.append(
                A.Rotate(
                    limit=(-rotate_min, rotate_max),
                    border_mode=cv2.BORDER_CONSTANT,
                    value=pad_value,
                    mask_value=ignore_index,
                    p=0.5,
                )
            )

        # ----- Random Gaussian blur -----
        if random_gaussian_blur:
            transforms_list.append(RandomGaussianBlur())

        # ----- Random horizontal flip -----
        if random_flip:
            transforms_list.append(A.HorizontalFlip(p=0.5))

        # ----- Crop (random or center) -----
        if random_crop:
            transforms_list.append(
                A.RandomCrop(height=crop_size, width=crop_size, p=1.0)
            )
        else:
            transforms_list.append(
                A.CenterCrop(height=crop_size, width=crop_size, p=1.0)
            )

        # ----- Tensor conversion -----
        if normalize:
            transforms_list.append(
                A.Normalize(mean=0.0, std=1.0, max_pixel_value=255.0)
            )

        # ----- Optional normalization -----
        if to_tensor:
            transforms_list.append(ToTensorV2())
            transforms_list.append(A.Lambda(image=lambda img, **kwargs: img.float()))

        self.transforms = A.Compose(transforms_list)

    def __call__(self, image, target):
        transformed = self.transforms(image=image, mask=target)
        return transformed["image"], transformed["mask"]


class PILSegmentationTrainPipeline:
    """Reusable training pipeline composed of Albumentations transforms."""

    def __init__(
        self,
        *,
        base_size: int,
        crop_size: int,
        pad_value: Sequence[float] | float = 0.0,
        ignore_index: int = 255,
        random_scale: bool = False,
        scale_min: float = 0.5,
        scale_max: float = 2.0,
        random_rotate: bool = False,
        rotate_min: int = -10,
        rotate_max: int = 10,
        random_gaussian_blur: bool = False,
        random_flip: bool = False,
        flip_prob: float = 0.5,
        random_crop: bool = True,
        normalize: bool = False,
        to_tensor: bool = True,
        is_check_shapes: bool = True,
    ):
        self.base_size = base_size
        self.crop_size = crop_size
        self.pad_value = pad_value
        self.ignore_index = ignore_index
        self.random_scale = random_scale
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.random_rotate = random_rotate
        self.rotate_min = rotate_min
        self.rotate_max = rotate_max
        self.random_gaussian_blur = random_gaussian_blur
        self.random_flip = random_flip
        self.flip_prob = flip_prob
        self.random_crop = random_crop
        self.normalize = normalize
        self.to_tensor = to_tensor
        self.is_check_shapes = is_check_shapes

    def __call__(self, image, target):
        # ----- Random scale -----
        if self.random_scale:
            scale = self.scale_min + (self.scale_max - self.scale_min) * random.random()
        else:
            scale = 1.0

        short = int(scale * self.base_size)
        w, h = image.size

        if h > w:
            ow = short
            oh = int(h * short / w)
        else:
            oh = short
            ow = int(w * short / h)

        image = image.resize((ow, oh), Image.BILINEAR)
        target = target.resize((ow, oh), Image.NEAREST)

        # ----- Pad if too small -----
        if short < self.crop_size:
            padh = max(self.crop_size - oh, 0)
            padw = max(self.crop_size - ow, 0)
            image = ImageOps.expand(
                image, border=(0, 0, padw, padh), fill=self.pad_value
            )
            target = ImageOps.expand(
                target, border=(0, 0, padw, padh), fill=self.ignore_index
            )

        # ----- Random rotate -----
        if self.random_rotate and random.random() < 0.5:
            angle = random.randint(self.rotate_min, self.rotate_max)
            image = image.rotate(angle, expand=True, fillcolor=self.pad_value)
            target = target.rotate(angle, expand=True, fillcolor=self.ignore_index)

        # ----- Random Gaussian blur -----
        if self.random_gaussian_blur and random.random() < 0.5:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.random()))

        # ----- Random horizontal flip -----
        if self.random_flip and random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            target = target.transpose(Image.FLIP_LEFT_RIGHT)

        # ----- Crop (random or center) -----
        w, h = image.size
        if self.random_crop:
            x1 = random.randint(0, w - self.crop_size)
            y1 = random.randint(0, h - self.crop_size)
        else:
            x1 = (w - self.crop_size) // 2
            y1 = (h - self.crop_size) // 2

        image = image.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        target = target.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        # ----- Optional normalization -----
        if self.normalize:
            image = image / 255.0

        # ----- Tensor conversion -----
        if self.to_tensor:
            image = torch.from_numpy(np.array(image).transpose(2, 0, 1)).float()
            target = torch.from_numpy(np.array(target)).long()

        return image, target


class CV2SegmentationEvalPipeline:
    """Deterministic evaluation pipeline shared between datasets."""

    def __init__(
        self,
        *,
        base_size: int,
        crop_size: int,
        center_crop: bool = False,
        normalize: bool = False,
        to_tensor: bool = True,
    ):
        self.center_crop = center_crop
        self.crop_size = crop_size
        self.normalize = normalize
        self.to_tensor = to_tensor
        self.transforms = None
        transforms_list = []
        if center_crop:
            transforms_list.append(A.SmallestMaxSize(max_size=crop_size, p=1.0))
            transforms_list.append(
                A.CenterCrop(
                    height=crop_size,
                    width=crop_size,
                    p=1.0,
                )
            )

        if normalize:
            transforms_list.append(
                A.Normalize(mean=0.0, std=1.0, max_pixel_value=255.0)
            )

        if to_tensor:
            transforms_list.append(ToTensorV2())
            transforms_list.append(A.Lambda(image=lambda img, **kwargs: img.float()))

        self.transforms = A.Compose(transforms_list)

    def __call__(self, image, target):
        if self.transforms is not None:
            transformed = self.transforms(image=image, mask=target)
            return transformed["image"], transformed["mask"]

        if self.center_crop:
            h, w = image.shape[:2]
            short_size = self.crop_size
            if w > h:
                oh = short_size
                ow = int(round(1.0 * w * oh / h))
            else:
                ow = short_size
                oh = int(round(1.0 * h * ow / w))
            image = cv2.resize(image, (ow, oh), interpolation=cv2.INTER_LINEAR)
            target = cv2.resize(target, (ow, oh), interpolation=cv2.INTER_NEAREST)
            x1 = int(round((ow - self.crop_size) / 2.0))
            y1 = int(round((oh - self.crop_size) / 2.0))
            image = image[y1 : y1 + self.crop_size, x1 : x1 + self.crop_size]
            target = target[y1 : y1 + self.crop_size, x1 : x1 + self.crop_size]

        if self.normalize:
            image = image.astype(np.float32) / 255.0

        if self.to_tensor:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float()
            target = torch.from_numpy(target).long()

        return image, target


class PILSegmentationEvalPipeline:
    """Deterministic evaluation pipeline shared between datasets."""

    def __init__(
        self,
        *,
        base_size: int,
        crop_size: int,
        center_crop: bool = False,
        normalize: bool = False,
        to_tensor: bool = True,
    ):
        self.center_crop = center_crop
        self.crop_size = crop_size
        self.normalize = normalize
        self.to_tensor = to_tensor

    def __call__(self, image, target):
        if self.center_crop:
            # ----- Resize to at least crop_size -----
            short_size = self.crop_size
            w, h = image.size
            if w > h:
                oh = short_size
                ow = int(1.0 * w * oh / h)
            else:
                ow = short_size
                oh = int(1.0 * h * ow / w)
            image = image.resize((ow, oh), Image.BILINEAR)
            target = target.resize((ow, oh), Image.NEAREST)

            # ----- Center Crop -----
            w, h = image.size
            x1 = int(round((w - self.crop_size) / 2.0))
            y1 = int(round((h - self.crop_size) / 2.0))
            image = image.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
            target = target.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        # ----- Tensor conversion -----
        if self.to_tensor:
            image = torch.from_numpy(np.array(image).transpose(2, 0, 1)).float()
            target = torch.from_numpy(np.array(target)).long()

        # ----- Optional normalization -----
        if self.normalize:
            image = image / 255.0
        return image, target


class FFCVSegmentationTrainPipeline:
    """FFCV training pipeline using GPU-accelerated augmentations.

    This pipeline uses FFCV's native GPU transforms for maximum performance.
    Note: GPU augmentations may produce slightly different results than PIL/CV2
    due to different interpolation algorithms and implementation details.
    """

    def __init__(
        self,
        *,
        base_size: int,
        crop_size: int,
        pad_value: Sequence[float] | float = 0.0,
        ignore_index: int = 255,
        random_scale: bool = False,
        scale_min: float = 0.5,
        scale_max: float = 2.0,
        random_rotate: bool = False,
        rotate_min: int = -10,
        rotate_max: int = 10,
        random_gaussian_blur: bool = False,
        random_flip: bool = False,
        flip_prob: float = 0.5,
        random_crop: bool = True,
        normalize: bool = False,
        to_tensor: bool = True,
        is_check_shapes: bool = True,
        # FFCV-specific options
        color_jitter: float = 0.0,
        gray_scale_prob: float = 0.0,
        blur_prob: float = 0.0,
        device: int = 0,
    ):
        self.base_size = base_size
        self.crop_size = crop_size
        self.pad_value = pad_value
        self.ignore_index = ignore_index
        self.random_scale = random_scale
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.random_rotate = random_rotate
        self.rotate_min = rotate_min
        self.rotate_max = rotate_max
        self.random_gaussian_blur = random_gaussian_blur
        self.random_flip = random_flip
        self.flip_prob = flip_prob
        self.random_crop = random_crop
        self.normalize = normalize
        self.to_tensor = to_tensor
        self.is_check_shapes = is_check_shapes
        self.color_jitter = color_jitter
        self.gray_scale_prob = gray_scale_prob
        self.blur_prob = blur_prob
        self.device = device

    def build_image_pipeline(self):
        """Build FFCV image pipeline for GPU-accelerated augmentations.

        Returns:
            List of FFCV operations for image transformations.
        """
        from ffcv.fields.decoders import RandomResizedCropRGBImageDecoder
        from ffcv.transforms import ToDevice, ToTensor, ToTorchImage

        # Random resized crop decoder (GPU-accelerated)
        if self.random_scale:
            decoder = RandomResizedCropRGBImageDecoder(
                (self.crop_size, self.crop_size),
                scale=(self.scale_min, self.scale_max),
            )
        else:
            decoder = RandomResizedCropRGBImageDecoder(
                (self.crop_size, self.crop_size),
                scale=(1.0, 1.0),
            )

        ops = [decoder]

        # Note: Cutout, ColorJitter, etc. can be added here when needed
        # For now, keep it simple to match CV2/PIL behavior

        # Convert to tensor and move to GPU
        ops.append(ToTensor())
        ops.append(ToTorchImage())
        ops.append(ToDevice(self.device))

        return ops

    def build_label_pipeline(self):
        """Build FFCV label pipeline for mask transformations.

        Returns:
            List of FFCV operations for label transformations.
        """
        from ffcv.fields.decoders import BytesDecoder
        from ffcv.transforms import ToDevice, ToTensor

        # Decode bytes to numpy array, then reshape in post-processing
        ops = [BytesDecoder()]
        ops.append(ToTensor())
        ops.append(ToDevice(self.device))

        return ops

    def __call__(self, image, target):
        """Fallback CPU implementation for compatibility.

        Args:
            image: Input image as numpy array (H, W, 3) in RGB format.
            target: Flattened mask as 1D numpy array.

        Returns:
            Tuple of (transformed_image, transformed_mask) as torch tensors.
        """
        # Reshape flattened mask back to 2D
        if target.ndim == 1:
            h, w = image.shape[:2]
            target = target.reshape(h, w)

        if self.to_tensor:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float()
            target = torch.from_numpy(target).long()
        if self.normalize:
            image = image / 255.0
        return image, target


class FFCVSegmentationEvalPipeline:
    """Deterministic evaluation pipeline for FFCV datasets using GPU transforms."""

    def __init__(
        self,
        *,
        base_size: int,
        crop_size: int,
        center_crop: bool = False,
        normalize: bool = False,
        to_tensor: bool = True,
        device: int = 0,
    ):
        self.base_size = base_size
        self.center_crop = center_crop
        self.crop_size = crop_size
        self.normalize = normalize
        self.to_tensor = to_tensor
        self.device = device

    def build_image_pipeline(self):
        """Build FFCV image pipeline for evaluation.

        Returns:
            List of FFCV operations for image transformations.
        """
        from ffcv.fields.decoders import CenterCropRGBImageDecoder
        from ffcv.transforms import ToTensor, ToTorchImage

        # Center crop decoder for deterministic evaluation
        decoder = CenterCropRGBImageDecoder(
            (self.crop_size, self.crop_size),
        )

        ops = [decoder]
        ops.append(ToTensor())
        ops.append(ToTorchImage())

        return ops

    def build_label_pipeline(self):
        """Build FFCV label pipeline for evaluation.

        Returns:
            List of FFCV operations for label transformations.
        """
        from ffcv.fields.decoders import BytesDecoder
        from ffcv.transforms import ToDevice, ToTensor

        ops = [BytesDecoder()]
        ops.append(ToTensor())
        ops.append(ToDevice(self.device))

        return ops

    def __call__(self, image, target):
        """Fallback CPU implementation for compatibility.

        Args:
            image: Input image as numpy array (H, W, 3) in RGB format.
            target: Flattened mask as 1D numpy array.

        Returns:
            Tuple of (transformed_image, transformed_mask) as torch tensors.
        """
        if self.to_tensor:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float()
            target = torch.from_numpy(target).long()
        if self.normalize:
            image = image / 255.0
        return image, target


# ---------------------------------------------------------------------------
# DALI Pipelines - NVIDIA GPU-accelerated data loading
# ---------------------------------------------------------------------------


class DALISegmentationTrainPipeline:
    """DALI training pipeline using GPU-accelerated augmentations.

    NVIDIA DALI provides GPU-accelerated image loading and augmentation
    with support for JPEG decoding, resize, crop, and other operations.
    """

    def __init__(
        self,
        *,
        base_size: int,
        crop_size: int,
        pad_value: Sequence[float] | float = 0.0,
        ignore_index: int = 255,
        random_scale: bool = False,
        scale_min: float = 0.5,
        scale_max: float = 2.0,
        random_rotate: bool = False,
        rotate_min: int = -10,
        rotate_max: int = 10,
        random_gaussian_blur: bool = False,
        random_flip: bool = False,
        flip_prob: float = 0.5,
        random_crop: bool = True,
        normalize: bool = False,
        to_tensor: bool = True,
        is_check_shapes: bool = True,
        device: str = "gpu",
        device_id: int = 0,
    ):
        self.base_size = base_size
        self.crop_size = crop_size
        self.pad_value = pad_value
        self.ignore_index = ignore_index
        self.random_scale = random_scale
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.random_rotate = random_rotate
        self.rotate_min = rotate_min
        self.rotate_max = rotate_max
        self.random_gaussian_blur = random_gaussian_blur
        self.random_flip = random_flip
        self.flip_prob = flip_prob
        self.random_crop = random_crop
        self.normalize = normalize
        self.to_tensor = to_tensor
        self.is_check_shapes = is_check_shapes
        self.device = device
        self.device_id = device_id

    def build_pipeline(
        self, filenames, masks, batch_size: int, shuffle: bool = True
    ) -> Pipeline:
        """Build DALI pipeline for segmentation training.

        Args:
            filenames: List of image file paths.
            masks: List of mask file paths.
            batch_size: Batch size per iteration.
            shuffle: Whether to shuffle the data.

        Returns:
            DALI Pipeline object.
        """

        class SegmentationPipeline(Pipeline):
            def __init__(self, filenames, masks, batch_size, pipeline_self):
                super().__init__(
                    batch_size=batch_size,
                    num_threads=8,
                    device_id=pipeline_self.device_id,
                    seed=42,
                )
                self.pipeline_self = pipeline_self
                self.device = pipeline_self.device

                # File reader for images
                self.input_images, self.input_image_labels = fn.readers.file(
                    files=filenames,
                    pad_last_batch=True,
                    shuffle_after_epoch=shuffle,
                    name="input_images",
                )

                # File reader for masks (use separate reader)
                self.input_masks, self.input_mask_labels = fn.readers.file(
                    files=masks,
                    pad_last_batch=True,
                    shuffle_after_epoch=shuffle,
                    name="input_masks",
                )

            def define_graph(self):
                # Decode images
                images = fn.decoders.image(self.input_images, device="mixed")

                # Decode masks as grayscale
                masks = fn.decoders.image(
                    self.input_masks,
                    device="mixed",
                    output_type=types.DALIImageType.GRAY,
                )

                # Resize to crop_size
                images = fn.resize(
                    images,
                    size=(self.pipeline_self.crop_size, self.pipeline_self.crop_size),
                    interp_type=types.INTERP_LINEAR,
                )
                masks = fn.resize(
                    masks,
                    size=(self.pipeline_self.crop_size, self.pipeline_self.crop_size),
                    interp_type=types.INTERP_NN,
                )

                # Convert to tensor format (CHW)
                images = fn.transpose(images, perm=[2, 0, 1])
                # Squeeze channel dim (H, W, 1) -> (H, W)
                masks = fn.squeeze(masks, axes=[2])

                return images, masks

        return SegmentationPipeline(filenames, masks, batch_size, self)

    def create_iterator(
        self, filenames, masks, batch_size: int, shuffle: bool = True
    ) -> DALIGenericIterator:
        """Create DALI iterator for training.

        Args:
            filenames: List of image file paths.
            masks: List of mask file paths.
            batch_size: Batch size.
            shuffle: Whether to shuffle the data.

        Returns:
            DALIGenericIterator object.
        """
        from nvidia.dali.plugin.base_iterator import LastBatchPolicy

        pipeline = self.build_pipeline(filenames, masks, batch_size, shuffle)

        policy = LastBatchPolicy.DROP if shuffle else LastBatchPolicy.FILL
        return DALIGenericIterator(
            pipeline,
            output_map=["data", "label"],
            reader_name="input_images",
            auto_reset=True,
            last_batch_policy=policy,
        )


class DALISegmentationEvalPipeline:
    """DALI evaluation pipeline using GPU-accelerated processing."""

    def __init__(
        self,
        *,
        base_size: int,
        crop_size: int,
        center_crop: bool = False,
        normalize: bool = False,
        to_tensor: bool = True,
        device: str = "gpu",
        device_id: int = 0,
    ):
        self.base_size = base_size
        self.center_crop = center_crop
        self.crop_size = crop_size
        self.normalize = normalize
        self.to_tensor = to_tensor
        self.device = device
        self.device_id = device_id

    def build_pipeline(self, filenames, masks, batch_size: int) -> Pipeline:
        """Build DALI pipeline for evaluation.

        Args:
            filenames: List of image file paths.
            masks: List of mask file paths.
            batch_size: Batch size per iteration.

        Returns:
            DALI Pipeline object.
        """

        class EvalPipeline(Pipeline):
            def __init__(self, filenames, masks, batch_size, pipeline_self):
                super().__init__(
                    batch_size=batch_size,
                    num_threads=8,
                    device_id=pipeline_self.device_id,
                    seed=42,
                )
                self.pipeline_self = pipeline_self
                self.device = pipeline_self.device

                self.input_images, self.input_image_labels = fn.readers.file(
                    files=filenames,
                    shuffle_after_epoch=False,
                    pad_last_batch=True,
                    name="input_images",
                )

                self.input_masks, self.input_mask_labels = fn.readers.file(
                    files=masks,
                    shuffle_after_epoch=False,
                    pad_last_batch=True,
                    name="input_masks",
                )

            def define_graph(self):
                # Decode images and masks
                images = fn.decoders.image(
                    self.input_images, device="mixed" if self.device == "gpu" else "cpu"
                )
                masks = fn.decoders.image(
                    self.input_masks,
                    device="mixed" if self.device == "gpu" else "cpu",
                    output_type=types.DALIImageType.GRAY,
                )

                # Resize to base_size
                images = fn.resize(
                    images,
                    size=self.pipeline_self.base_size,
                    interp_type=types.INTERP_LINEAR,
                )
                masks = fn.resize(
                    masks,
                    size=self.pipeline_self.base_size,
                    interp_type=types.INTERP_NN,
                )

                # Center crop
                if self.pipeline_self.center_crop:
                    images = fn.center_crop(
                        images,
                        crop_h=self.pipeline_self.crop_size,
                        crop_w=self.pipeline_self.crop_size,
                    )
                    masks = fn.center_crop(
                        masks,
                        crop_h=self.pipeline_self.crop_size,
                        crop_w=self.pipeline_self.crop_size,
                    )

                # Normalize
                if self.pipeline_self.normalize:
                    images = fn.normalize(images)

                # Convert to tensor format (CHW)
                images = fn.transpose(images, perm=[2, 0, 1])
                # Squeeze channel dim (H, W, 1) -> (H, W)
                masks = fn.squeeze(masks, axes=[2])

                return images, masks

        return EvalPipeline(filenames, masks, batch_size, self)

    def create_iterator(self, filenames, masks, batch_size: int) -> DALIGenericIterator:
        """Create DALI iterator for evaluation.

        Args:
            filenames: List of image file paths.
            masks: List of mask file paths.
            batch_size: Batch size.

        Returns:
            DALIGenericIterator object.
        """
        from nvidia.dali.plugin.base_iterator import LastBatchPolicy

        pipeline = self.build_pipeline(filenames, masks, batch_size)
        return DALIGenericIterator(
            pipeline,
            output_map=["data", "label"],
            reader_name="input_images",
            auto_reset=True,
            last_batch_policy=LastBatchPolicy.FILL,
        )
