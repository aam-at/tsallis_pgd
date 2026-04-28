from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from copy import deepcopy
from io import StringIO
from typing import Any

import numpy as np
import torch
from torcheval.metrics import Mean
from torcheval.metrics.toolkit import get_synced_metric_collection


class NanError(BaseException):
    def __init__(self, message=None):
        self.message = message

    def __str__(self):
        return self.message


class MeanMetricsDictionary(dict[str, Mean]):
    """Dictionary that lazily creates torcheval mean metrics by key."""

    def __init__(
        self,
        *args: Any,
        device: torch.device | str | None = None,
        sync_on_compute: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.device = torch.device(device) if device is not None else None
        self.sync_on_compute = sync_on_compute

    def __missing__(self, key: str) -> Mean:
        metric = Mean(device=self.device) if self.device is not None else Mean()
        self[key] = metric
        return metric

    def set_sync_on_compute(self, sync_on_compute: bool) -> MeanMetricsDictionary:
        self.sync_on_compute = sync_on_compute
        return self

    def to(self, device: torch.device | str) -> MeanMetricsDictionary:
        self.device = torch.device(device)
        for metric in self.values():
            metric.to(self.device)
        return self

    def sync(self) -> MeanMetricsDictionary:
        synced_metrics = MeanMetricsDictionary(device=self.device)
        active_metrics = self.active()
        if not active_metrics:
            return synced_metrics

        for name, metric in get_synced_metric_collection(active_metrics).items():
            synced_metrics[name] = deepcopy(metric)
        return synced_metrics

    def active(self) -> dict[str, Mean]:
        return {
            name: metric for name, metric in self.items() if _metric_has_data(metric)
        }

    def reset(self) -> None:
        for metric in self.values():
            metric.reset()


def _metric_weight(metric: Any) -> torch.Tensor | None:
    if hasattr(metric, "weights"):
        return metric.weights
    if hasattr(metric, "count"):
        return metric.count
    return None


def _metric_has_data(metric: Any) -> bool:
    weight = _metric_weight(metric)
    if weight is None:
        return True
    if torch.is_tensor(weight):
        return bool(weight.detach().item())
    return bool(weight)


def format_float(v, digits=8):
    """Format float value and strip zeros."""
    v = np.round(v, digits)
    v_fmt = (f"%.{digits}f" % v).rstrip("0").rstrip(".")
    return v_fmt


def metrics_to_dict(
    metrics: Mapping[str, Any],
    prefix: str = "",
    *,
    sync: bool | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Convert metrics to dict."""
    if sync is None and isinstance(metrics, MeanMetricsDictionary):
        sync = metrics.sync_on_compute

    if sync and isinstance(metrics, MeanMetricsDictionary):
        active_metrics = metrics.sync().active()
    else:
        active_metrics = {
            name: metric for name, metric in metrics.items() if _metric_has_data(metric)
        }

    computed_metrics: Mapping[str, Any]
    computed_metrics = {
        name: metric.compute() for name, metric in active_metrics.items()
    }

    d: dict[str, Any] = {}
    nan_metrics: list[str] = []
    for name, metric_value in computed_metrics.items():
        if torch.is_tensor(metric_value):
            metric_value = float(metric_value.detach().item())

        metric_name = f"{prefix}{name}"
        if math.isnan(metric_value):
            nan_metrics.append(metric_name)
            d[metric_name] = 1e6
        else:
            d[metric_name] = metric_value
    return d, nan_metrics


def reset_metrics(metrics: Mapping[str, Any]) -> None:
    for metric in metrics.values():
        metric.reset()


def log_metrics(
    metrics: Mapping[str, Any],
    *,
    prefix: str = "",
    header: str | None = "",
    throw_on_nan: bool = False,
    mode: str | None = None,
    use_logger: bool = True,
    use_tb: bool = False,
    use_wandb: bool = False,
    epoch: int | None = None,
    step: int | None = None,
) -> dict[str, float]:
    del mode, use_tb, use_wandb, epoch, step
    reduced_metrics, nan_metrics = metrics_to_dict(metrics, prefix=prefix)
    if use_logger and reduced_metrics:
        logging.info(format_dict_as_string(reduced_metrics, header=header))
    if nan_metrics and throw_on_nan:
        raise NanError(f"Nan was detected for metrics {', '.join(nan_metrics)}")
    return reduced_metrics


def format_dict_as_string(d: dict, header: str | None = "", digits: int = 6) -> str:
    """Format dict as string."""
    str_bfr = StringIO()
    if header is not None:
        str_bfr.write(f"{header}: ")
    for name, value in d.items():
        str_bfr.write(f" {name}: {format_float(value, digits)},")
    return str_bfr.getvalue()[:-1]
