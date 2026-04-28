from __future__ import absolute_import, division, print_function

from collections.abc import Sequence

import torch


def assert_none(**kwargs):
    """Assert that variable is not None."""
    assert len(kwargs) == 1
    k, v = next(iter(kwargs.items()))
    assert v is None, f"'{k}' must be 'None', given '{v}'"


def assert_not_none(**kwargs):
    """Assert that variable is not None."""
    assert len(kwargs) == 1
    k, v = next(iter(kwargs.items()))
    assert v is not None, f"'{k}' can't be '{v}'"


def assert_greater(vv, **kwargs):
    """Assert that variable is greater than v."""
    assert len(kwargs) == 1
    k, v = next(iter(kwargs.items()))
    if torch.is_tensor(v):
        assert (v > vv).all(), f"All elements in '{k}' must be greater than '{vv}'"
    elif isinstance(v, Sequence):
        for v_ in v:
            assert v_ > vv, f"All elements in '{k}' ({v}) must be greater than '{vv}'"
    else:
        assert v > vv, f"'{k}' ({v}) must be greater than '{vv}'"


def assert_greater_equal(vv, **kwargs):
    """Assert that variable is greater than v."""
    assert len(kwargs) == 1
    k, v = next(iter(kwargs.items()))
    if torch.is_tensor(v):
        assert (v >= vv).all(), (
            f"All elements in '{k}' must be greater or equal than '{vv}'"
        )
    elif isinstance(v, Sequence):
        for v_ in v:
            assert v_ >= vv, (
                f"All elements in '{k}' ({v}) must be greater or equal than '{vv}'"
            )
    else:
        assert v >= vv, f"'{k}' ({v}) must be greater or equal than '{vv}'"


def assert_less(vv, **kwargs):
    """Assert that variable is less than v."""
    assert len(kwargs) == 1
    k, v = next(iter(kwargs.items()))
    if torch.is_tensor(v):
        assert (v < vv).all(), f"All elements in '{k}' must be less than '{vv}'"
    elif isinstance(v, Sequence):
        for v_ in v:
            assert v_ < vv, f"All elements in '{k}' ({v}) must be less than '{vv}'"
    else:
        assert v < vv, f"'{k}' ({v}) must be less than '{vv}'"


def assert_less_equal(vv, **kwargs):
    """Assert that variable is less than v."""
    assert len(kwargs) == 1
    k, v = next(iter(kwargs.items()))
    if torch.is_tensor(v):
        assert (v <= vv).all(), (
            f"All elements in '{k}' must be less or equal than '{vv}'"
        )
    elif isinstance(v, Sequence):
        for v_ in v:
            assert v_ <= vv, (
                f"All elements in '{k}' ({v}) must be less or equal than '{vv}'"
            )
    else:
        assert v <= vv, f"'{k}' ({v}) must be less or equal than '{vv}'"


def assert_not_empty(**kwargs):
    """Assert that variable is not empty iterable."""
    assert len(kwargs) == 1
    k, v = next(iter(kwargs.items()))
    if torch.is_tensor(v):
        assert v.numel() > 0, f"'{k}' can't be empty"
    else:
        assert len(v) > 0, f"'{k}' can't be empty"


def assert_in_enum(allowed, **kwargs):
    """Assert that variable is in the list of allowed values."""
    assert len(kwargs) == 1
    k, v = next(iter(kwargs.items()))
    has_match = False
    for l_ in allowed:
        # Use simple string comparison instead of fnmatch for torch.compile compatibility
        if v == l_ or (isinstance(v, str) and isinstance(l_, str) and v in l_):
            has_match = True
            break
    assert has_match, f"'{k}={v} is not in the allowed values."


def assert_same_length(**kwargs):
    """Assert that all variables have the same length."""
    assert len(kwargs) > 1
    k, v = next(iter(kwargs.items()))
    if isinstance(v, Sequence) or torch.is_tensor(v):
        is_sequence = True
        length = len(v)
    else:
        is_sequence = False
    for k_, v_ in kwargs.items():
        if is_sequence:
            assert isinstance(v_, Sequence) or torch.is_tensor(v_), (
                f"'{k_}' must be a sequence or tensor."
            )
            assert len(v_) == length, f"'{k}' and '{k_}' must have the same length."
        else:
            assert not (isinstance(v_, Sequence) or torch.is_tensor(v_)), (
                f"'{k_}' must be a scalar."
            )


def assert_in_range(min_val, max_val, **kwargs):
    """Assert that variable is in the range [min_val, max_val]."""
    assert len(kwargs) == 1
    k, v = next(iter(kwargs.items()))
    if torch.is_tensor(v):
        assert (v >= min_val).all() and (v <= max_val).all(), (
            f"All elements in '{k}' must be in range [{min_val}, {max_val}]"
        )
    elif isinstance(v, Sequence):
        for v_ in v:
            assert min_val <= v_ <= max_val, (
                f"All elements in '{k}' ({v}) must be in range [{min_val}, {max_val}]"
            )
    else:
        assert min_val <= v <= max_val, (
            f"'{k}' ({v}) must be in range [{min_val}, {max_val}]"
        )


def assert_rank(expected_rank, **kwargs):
    """Assert that tensor has the expected rank."""
    assert len(kwargs) == 1
    k, v = next(iter(kwargs.items()))
    assert v.ndim == expected_rank, (
        f"'{k}' must have rank {expected_rank}, got {v.ndim}."
    )
