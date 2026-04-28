from collections.abc import Sequence
from typing import Any, Tuple

import torch
import torch.nn.functional as F
from segmentation_attacks.utils.types import Tensor, TensorShape


def batch_nanquantile(x: Tensor, q: Tensor, dim: int = -1) -> Tensor:
    """Per-sample quantile along `dim`, ignoring NaNs.

    `q` is a scalar per sample (broadcastable to x.shape with the reduction dim removed).
    Returns the same shape as x with `dim` removed.
    """
    # Move target dim to the end for simpler indexing
    x = torch.movedim(x, dim, -1)  # [..., M]
    x_dtype = x.dtype

    # NaN-safe sort: push NaNs to the end
    mask = torch.isnan(x)
    x_masked = x.clone()
    x_masked[mask] = float("inf")
    x_sorted, _ = torch.sort(x_masked, dim=-1)  # [..., M]

    # Count valid entries (non-NaN) per sample
    valid_counts = (~mask).sum(dim=-1)  # [...], int64

    # Compute fractional indices in float
    n_float = (valid_counts - 1).clamp(min=0).to(x_dtype)  # [...], float
    q_b = q.to(x_dtype)
    # Make sure q broadcasts to [...]; if q has the reduction dim included, squeeze it
    # (Most users pass q shaped like x.shape with the reduction dim removed.)
    idx = (q_b * n_float).clamp(
        min=torch.zeros_like(n_float), max=n_float
    )  # [...], float

    # Interpolate between floor/ceil ranks
    idx_floor = torch.floor(idx).long()  # [...]
    idx_ceil = torch.minimum(
        idx_floor + 1, (valid_counts - 1).clamp(min=0)
    )  # [...], long
    w = idx - idx_floor.to(x_dtype)  # [...], float in [0,1]

    # Gather needs index tensors with the gather dimension present
    idx_floor_g = idx_floor.unsqueeze(-1)  # [..., 1]
    idx_ceil_g = idx_ceil.unsqueeze(-1)  # [..., 1]
    x_floor = x_sorted.gather(-1, idx_floor_g).squeeze(-1)  # [...]
    x_ceil = x_sorted.gather(-1, idx_ceil_g).squeeze(-1)  # [...]

    out = (1 - w) * x_floor + w * x_ceil  # [...]

    # Move dims back: we reduced the last dim, so just return
    return out


def batch_update(tensors, updates, indices, in_place=True, select_updates=True):
    """Batch update of tensors in PyTorch.

    Args:
        tensors: A tensor or sequence of tensors to be updated.
        updates: A tensor or sequence of tensors with the update values.
        indices: A 1D tensor containing the indices to be updated.
        select_updates: If True, the updates tensor will be indexed using the indices.

    Returns:
        Updated tensor or sequence of tensors.
    """
    if isinstance(tensors, dict):
        return {
            key: batch_update(
                tensors[key],
                updates[key],
                indices,
                in_place=in_place,
                select_updates=select_updates,
            )
            for key in tensors
        }
    elif isinstance(tensors, Sequence):
        return type(tensors)(
            [
                batch_update(
                    t, u, indices, in_place=in_place, select_updates=select_updates
                )
                for t, u in zip(tensors, updates)
            ]
        )
    else:
        if not in_place:
            # Create a copy of the tensor to avoid in-place modification
            tensors = tensors.clone()
        if select_updates:
            updates = updates[indices].clone()
        # Update the tensor at the specified indices
        tensors[indices] = updates
        return tensors


def zeros_like(
    tensors: Tensor, override_shape: TensorShape = None
) -> Tensor | list[Tensor | Any] | dict[Any, Tensor | Any]:
    """Return a tensor of zeros with the same shape and type as the input
    tensor."""
    if isinstance(tensors, torch.Tensor):
        if override_shape is None:
            return torch.zeros_like(tensors)
        assert tensors.ndim == len(override_shape), (
            f"Tensor has {tensors.ndim} dimensions, but override_shape has {len(override_shape)} dimensions."
        )
        shape = []
        for i in range(tensors.dim()):
            if override_shape[i] is not None:
                shape.append(override_shape[i])
            else:
                shape.append(tensors.shape[i])
        return torch.zeros(shape, dtype=tensors.dtype, device=tensors.device)
    elif isinstance(tensors, Sequence):
        return type(tensors)(
            [
                zeros_like(item, override_shape)
                for item in tensors
                if override_shape is None
                or isinstance(item, dict)
                or (isinstance(item, torch.Tensor) and item.ndim == len(override_shape))
            ]
        )
    elif isinstance(tensors, dict):
        return {
            key: zeros_like(value, override_shape)
            for key, value in tensors.items()
            if override_shape is None
            or isinstance(value, dict)
            or (isinstance(value, torch.Tensor) and value.ndim == len(override_shape))
        }
    else:
        raise TypeError(f"Unsupported type: {type(tensors)}")


def tensor_to(tensors: Any, device: str | torch.device, non_blocking: bool = False):
    if isinstance(tensors, torch.Tensor):
        return tensors.to(device, non_blocking=non_blocking)
    elif isinstance(tensors, Sequence):
        return type(tensors)([tensor_to(t, device, non_blocking) for t in tensors])
    elif isinstance(tensors, dict):
        return {
            key: tensor_to(value, device, non_blocking)
            for key, value in tensors.items()
        }
    else:
        raise TypeError(f"Unsupported type: {type(tensors)}")


def tensor_select(
    tensors: Any,
    region_slice: Sequence[slice],
):
    if isinstance(tensors, torch.Tensor):
        return tensors[region_slice]
    elif isinstance(tensors, Sequence):
        return type(tensors)([tensor_select(t, region_slice) for t in tensors])
    elif isinstance(tensors, dict):
        return {
            key: tensor_select(value, region_slice) for key, value in tensors.items()
        }
    else:
        raise TypeError(f"Unsupported type: {type(tensors)}")


def tensor_add(output: Any, update: Any, region_slice: Sequence[slice]) -> None:
    if isinstance(update, torch.Tensor):
        output[region_slice] += update
    elif isinstance(update, Sequence):
        assert len(output) == len(update), (
            f"Output and update lists must have the same length. Got {len(output)} and {len(update)}."
        )
        for output_i, update_i in zip(output, update):
            tensor_add(output_i, update_i, region_slice)
    elif isinstance(update, dict):
        for key in update:
            if key in output:
                tensor_add(output[key], update[key], region_slice)


def tensor_scale(output: Tensor, scale: Tensor, region_slice: Sequence[slice]) -> None:
    if isinstance(output, torch.Tensor):
        output[region_slice] *= scale
    elif isinstance(output, Sequence):
        for output_i in output:
            tensor_scale(output_i, scale, region_slice)
    elif isinstance(output, dict):
        for key in output:
            tensor_scale(output[key], scale, region_slice)
    else:
        raise TypeError(f"Unsupported type: {type(output)}")


def tensor_resize(tensors: Any, size: Tuple[int, int]) -> Any:
    """Resize nested tensor structures (Tensor, list, dict) to the given (H, W)
    size. Uses bilinear interpolation for float tensors, nearest for int
    tensors.

    Args:
        tensors: A tensor or structure of tensors (list/dict).
        size: Target spatial size (height, width).

    Returns:
        Resized structure with the same nesting as the input.
    """
    h, w = size

    if isinstance(tensors, torch.Tensor):
        if tensors.ndim != 4:
            raise ValueError(f"Expected 4D NCHW tensor, got shape {tensors.shape}")
        if tensors.shape[2] == h and tensors.shape[3] == w:
            return tensors

        mode = "bilinear" if torch.is_floating_point(tensors) else "nearest"
        align_corners = True if mode == "bilinear" else None
        return F.interpolate(
            tensors, size=(h, w), mode=mode, align_corners=align_corners
        )

    elif isinstance(tensors, Sequence):
        return type(tensors)([tensor_resize(item, size) for item in tensors])

    elif isinstance(tensors, dict):
        return {key: tensor_resize(value, size) for key, value in tensors.items()}

    else:
        raise TypeError(f"Unsupported type: {type(tensors)}")


def tensor_pad(
    image, padding, mode: str = "constant", value: Tensor | float | None = None
):
    """Implement a padding function that can handle both constant and tensor
    values padding."""
    if mode == "constant" and isinstance(value, torch.Tensor):
        if len(padding) != 4:
            raise ValueError(f"Padding must be a tuple of 4 integers, got {padding}")
        if len(value) != image.shape[1]:
            raise ValueError(
                f"Value tensor must have the same number of channels as the image, got {value.shape[1]} and {image.shape[1]}"
            )
        b, c, h, w = image.shape
        left, right, top, bottom = padding

        new_h = h + top + bottom
        new_w = w + left + right

        # Create padded tensor filled with channel-wise mean
        padded = torch.empty(
            (b, c, new_h, new_w), dtype=image.dtype, device=image.device
        )
        for i in range(c):
            padded[:, i].fill_(value[i])

        # Insert original image in the center
        padded[:, :, top : top + h, left : left + w] = image

        return padded
    else:
        return F.pad(image, padding, mode=mode, value=value)
