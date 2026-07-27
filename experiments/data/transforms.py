from typing import Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from numpy import random
from PIL import Image
from torchvision import transforms


class ResizeLongSide(A.DualTransform):
    """Resize image and mask so that the longer side equals `long_size`,
    preserving aspect ratio.

    Uses INTER_LINEAR for image, INTER_NEAREST for mask.
    """

    def __init__(self, long_size: int, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.long_size = long_size

    def apply(self, image, **params):
        h, w = image.shape[:2]
        new_h, new_w = self._get_new_hw(h, w)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    def apply_to_mask(self, mask, **params):
        h, w = mask.shape[:2]
        new_h, new_w = self._get_new_hw(h, w)
        return cv2.resize(
            mask.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST
        )

    def _get_new_hw(self, h, w):
        new_h, new_w = self.long_size, self.long_size
        if h > w:
            new_w = round(self.long_size / float(h) * w)
        else:
            new_h = round(self.long_size / float(w) * h)
        return new_h, new_w

    def get_transform_init_args_names(self):
        return ("long_size",)


class PhotoMetricDistortion(torch.nn.Module):
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.

    1. random brightness
    2. random contrast (mode 0)
    3. convert color from RGB to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to RGB
    7. random contrast (mode 1)

    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(
        self,
        brightness_delta: int = 32,
        contrast_range: Tuple[float, float] = (0.5, 1.5),
        saturation_range: Tuple[float, float] = (0.5, 1.5),
        hue_delta: int = 18,
    ) -> None:
        super().__init__()
        self.brightness_delta: int = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta: int = hue_delta

    def brightness(self, img: Image.Image) -> Image.Image:
        """Brightness distortion."""
        if random.random() < 0.5:
            delta = random.uniform(-self.brightness_delta, self.brightness_delta)
            return transforms.functional.adjust_brightness(img, 1 + delta / 255.0)
        return img

    def contrast(self, img: Image.Image) -> Image.Image:
        """Contrast distortion."""
        if random.random() < 0.5:
            alpha = random.uniform(self.contrast_lower, self.contrast_upper)
            return transforms.functional.adjust_contrast(img, alpha)
        return img

    def saturation(self, img: Image.Image) -> Image.Image:
        """Saturation distortion."""
        if random.random() < 0.5:
            alpha = random.uniform(self.saturation_lower, self.saturation_upper)
            return transforms.functional.adjust_saturation(img, alpha)
        return img

    def hue(self, img: Image.Image) -> Image.Image:
        """Hue distortion."""
        if random.random() < 0.5:
            delta = random.uniform(-self.hue_delta, self.hue_delta)
            return transforms.functional.adjust_hue(img, delta / 180.0)
        return img

    def forward(self, img: Image.Image) -> Image.Image:
        """Transform function to perform photometric distortion on images."""
        # random brightness
        img = self.brightness(img)

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = random.randint(0, 1)
        if mode == 1:
            img = self.contrast(img)

        # Convert to HSV space
        img = img.convert("HSV")

        # random saturation
        img = self.saturation(img)

        # random hue
        img = self.hue(img)

        # Convert back to RGB
        img = img.convert("RGB")

        # random contrast
        if mode == 0:
            img = self.contrast(img)

        return img

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"brightness_delta={self.brightness_delta}, "
            f"contrast_range=({self.contrast_lower}, {self.contrast_upper}), "
            f"saturation_range=({self.saturation_lower}, {self.saturation_upper}), "
            f"hue_delta={self.hue_delta})"
        )
