from __future__ import annotations

from enum import Enum

import cv2
import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.utils import save_image
from utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class ArrayImage:
    def __init__(self, array: np.ndarray):
        self.array = np.asarray(array, dtype=np.uint8)

    def save(self, file_path: str) -> None:
        cv2.imwrite(file_path, cv2.cvtColor(self.array, cv2.COLOR_RGB2BGR))


class CombinedDataset(Dataset):
    def __init__(self, *datasets):
        # sanity check: all datasets must have same length
        self.datasets = datasets
        self.length = len(datasets[0])
        assert all(len(d) == self.length for d in datasets), (
            "Datasets have unequal length"
        )

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return tuple(d[idx] for d in self.datasets)


class DataSplit(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


def validate_enum_value(value: str, enum: Enum) -> None:
    if value not in [v.value for v in enum]:
        raise ValueError(f"Invalid value {value} for enum {enum}.")


def tensor_to_np_image(tensor: Tensor) -> np.ndarray:
    """Convert a PyTorch tensor to a np.ndarray Image."""
    nd_tensor = (
        tensor.mul(255)
        .add_(0.5)
        .clamp_(0, 255)
        .permute(1, 2, 0)
        .to("cpu", torch.uint8)
        .numpy()
    )
    return nd_tensor


def colorize(
    gray: np.ndarray | Tensor,
    palette: list[int] | list[tuple[int, int, int]] | np.ndarray,
    start_index: int | None = None,
) -> Image.Image:
    """Colorize a mask using a palette, handling label offsets (e.g., ADE20K
    starts at 1)."""
    pal = np.array(palette, dtype=np.uint8).reshape(-1, 3)

    gray_np = gray.detach().cpu().numpy() if isinstance(gray, torch.Tensor) else gray
    min_label = int(np.min(gray_np)) if gray_np.size > 0 else 0
    offset = min_label if start_index is None else start_index

    if offset and offset > 0:
        pal = np.vstack([np.zeros((offset, 3), dtype=np.uint8), pal])

    if Image is not None:
        palette_flat = pal.flatten().tolist()
        color = Image.fromarray(gray_np.astype(np.uint8)).convert("P")
        color.putpalette(palette_flat)
        return color

    rgb = pal[gray_np.astype(np.int64)]
    return ArrayImage(rgb)


def tensor_to_pil_image(tensor: Tensor) -> Image.Image:
    """Convert a PyTorch tensor to a PIL Image."""
    nd_tensor = tensor_to_np_image(tensor)
    return Image.fromarray(nd_tensor)


def pred_to_image(prediction: Tensor, cmap: Tensor) -> Tensor:
    """Convert a prediction tensor to an RGB image tensor.

    Args:
        prediction (Tensor): A tensor of shape (H, W) containing class indices.
        cmap (Tensor): A color map tensor of shape (num_classes, 3)
        containing RGB values for each class. If None, a default color map is
        used.

    Returns:
        Tensor: An RGB image tensor of shape (3, H, W) with values in [0, 1].
    """
    cmap = torch.as_tensor(cmap / 255, dtype=torch.float, device=prediction.device)
    pred_image = cmap[prediction].movedim(-1, -3)
    return pred_image


def save_pred(image: torch.Tensor, file_path: str) -> None:
    try:
        save_image(image, file_path)
        log.info(f"Image successfully saved to {file_path}")
    except Exception as e:
        log.error(f"Failed to save image: {e}")


def save_to_video(frames: list[Image.Image], output_file: str, fps: int = 30) -> None:
    """Save a list of PIL Image frames to a video file.

    Args:
        frames (list[Image.Image]): A list of PIL Image objects representing video frames.
        output_file (str): The path where the output video file will be saved.
        fps (int, optional): Frames per second for the output video. Defaults to 30.

    Returns:
        None
    """

    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(np.array(frame))

    out.release()
