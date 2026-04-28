"""Attack success metrics and registry-backed objective construction."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from .types import Loss


def _compute_confusion_matrix(
    logits: Tensor, labels: Tensor, ignore_index: int = -1
) -> Tensor:
    """Compute a per-sample confusion matrix from logits and labels."""

    batch_size, num_classes = logits.size(0), logits.size(1)
    pred = logits.argmax(1)
    valid = labels != ignore_index
    pred = torch.where(valid, pred, torch.full_like(pred, ignore_index))
    batch_indices = torch.arange(batch_size, device=logits.device).view(
        -1, *([1] * (labels.dim() - 1))
    )
    batch_indices = batch_indices.expand_as(labels)
    batch_offset = num_classes**2
    indices = (batch_indices * batch_offset + num_classes * labels.long() + pred)[valid]
    return (
        torch.bincount(indices, minlength=batch_size * num_classes**2)
        .reshape(batch_size, num_classes, num_classes)
        .float()
    )


def _error_rates_from_confusion(confusion: Tensor) -> tuple[Tensor, Tensor]:
    """Return global accuracy error and mean-class accuracy error."""

    true_positives = torch.diagonal(confusion, dim1=1, dim2=2)
    acc_global = true_positives.sum(1) / confusion.flatten(1).sum(1)
    acc_mean = (true_positives / confusion.sum(2)).nanmean(1)
    return 1.0 - acc_global, 1.0 - acc_mean


def attack_success(
    logits: Tensor, labels: Tensor, ignore_index: int = -1
) -> tuple[Tensor, Tensor]:
    """Compute per-sample global and mean-class attack success rates."""

    confusion = _compute_confusion_matrix(logits, labels, ignore_index=ignore_index)
    err_global, err_mean = _error_rates_from_confusion(confusion)
    return err_global, err_mean


def ce_objective(logits: Tensor, labels: Tensor, ignore_index: int) -> Tensor:
    """Compute per-sample cross-entropy objective."""

    loss = F.cross_entropy(logits, labels, ignore_index=ignore_index, reduction="none")
    return loss.flatten(1).mean(1)


def acc_global_objective(logits: Tensor, labels: Tensor, ignore_index: int) -> Tensor:
    """Compute per-sample global accuracy error."""

    err_global, _ = attack_success(logits, labels, ignore_index=ignore_index)
    return err_global


def acc_mean_objective(logits: Tensor, labels: Tensor, ignore_index: int) -> Tensor:
    """Compute per-sample mean-class accuracy error."""

    _, err_mean = attack_success(logits, labels, ignore_index=ignore_index)
    return err_mean


def _wrap_objective(fn: Loss, ignore_index: int) -> Loss:
    """Wrap an objective function to inject ``ignore_index``."""

    def wrapper(logits: Tensor, labels: Tensor) -> Tensor:
        return fn(logits, labels, ignore_index=ignore_index)

    return wrapper


class ObjectiveFunction:
    """Wrap a single objective callable."""

    def __init__(self, fn: Loss, name: str) -> None:
        self.fn = fn
        self.name = name

    def __call__(self, logits: Tensor, labels: Tensor) -> Tensor:
        return self.fn(logits, labels)

    def __repr__(self) -> str:
        return f"ObjectiveFunction({self.name!r})"


class CompositeObjectiveFunction(ObjectiveFunction):
    """Evaluate multiple objectives and return the element-wise maximum."""

    def __init__(self, fns: list[Loss], names: list[str]) -> None:
        self.fns = fns
        self.names = names
        super().__init__(fn=self._composite, name="+".join(names))

    def _composite(self, logits: Tensor, labels: Tensor) -> Tensor:
        values = torch.stack([fn(logits, labels) for fn in self.fns], dim=0)
        return values.max(dim=0).values

    def __repr__(self) -> str:
        return f"CompositeObjectiveFunction({self.names!r})"


@dataclass(frozen=True)
class ObjectiveBuildContext:
    """Inputs required to construct an objective callable."""

    ignore_index: int


ObjectiveBuilder = Callable[[ObjectiveBuildContext], Loss]


@dataclass(frozen=True)
class ObjectiveSpec:
    """Normalized objective configuration used by attacks."""

    names: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ObjectiveRegistration:
    """Registry entry describing a canonical objective and its aliases."""

    name: str
    builder: ObjectiveBuilder
    aliases: tuple[str, ...] = ()


class ObjectiveFactory:
    """Registry-backed factory for resolving and building objectives."""

    def __init__(self) -> None:
        self._entries: dict[str, ObjectiveRegistration] = {}
        self._aliases: dict[str, str] = {}

    def register(
        self, name: str, *, aliases: Sequence[str] = ()
    ) -> Callable[[ObjectiveBuilder], ObjectiveBuilder]:
        canonical = name.lower()
        normalized_aliases = tuple(alias.lower() for alias in aliases)

        def decorator(builder: ObjectiveBuilder) -> ObjectiveBuilder:
            self._register(
                ObjectiveRegistration(
                    name=canonical,
                    builder=builder,
                    aliases=normalized_aliases,
                )
            )
            return builder

        return decorator

    def _register(self, registration: ObjectiveRegistration) -> None:
        if registration.name in self._entries or registration.name in self._aliases:
            raise ValueError(f"Objective '{registration.name}' is already registered")
        for alias in registration.aliases:
            if alias == registration.name:
                continue
            if alias in self._entries or alias in self._aliases:
                raise ValueError(
                    f"Alias '{alias}' for objective '{registration.name}' is already registered"
                )
        self._entries[registration.name] = registration
        for alias in registration.aliases:
            if alias != registration.name:
                self._aliases[alias] = registration.name

    def supported_names(self, *, include_aliases: bool = True) -> tuple[str, ...]:
        names = list(self._entries)
        if include_aliases:
            names.extend(alias for alias in self._aliases if alias not in self._entries)
        return tuple(names)

    def normalize_name(self, objective: str) -> str:
        lowered = objective.lower()
        if lowered in self._entries:
            return lowered
        alias = self._aliases.get(lowered)
        if alias is not None:
            return alias
        raise ValueError(
            f"Unknown objective '{objective}'. Supported names: {sorted(self.supported_names())}"
        )

    def resolve(
        self,
        objective: str | Sequence[str] | Mapping[str, Any] | ObjectiveSpec,
    ) -> ObjectiveSpec:
        if isinstance(objective, ObjectiveSpec):
            return ObjectiveSpec(
                names=tuple(self.normalize_name(name) for name in objective.names)
            )
        if isinstance(objective, str):
            return ObjectiveSpec(names=(self.normalize_name(objective),))
        if isinstance(objective, Mapping):
            raw_names = objective.get("type", objective.get("types"))
            if raw_names is None:
                raise TypeError(
                    "objective mapping must contain a 'type' or 'types' key"
                )
            objective = raw_names
        if isinstance(objective, Sequence):
            names = tuple(self.normalize_name(name) for name in objective)
            if not names:
                raise ValueError("objective sequence must not be empty")
            return ObjectiveSpec(names=names)
        raise TypeError(
            f"objective spec must be str | Sequence[str] | Mapping | ObjectiveSpec, got {type(objective).__name__}"
        )

    def build(
        self,
        objective: str | Sequence[str] | Mapping[str, Any] | ObjectiveSpec,
        *,
        ignore_index: int,
    ) -> tuple[ObjectiveFunction, ObjectiveSpec]:
        spec = self.resolve(objective)
        context = ObjectiveBuildContext(ignore_index=ignore_index)
        fns = [self._entries[name].builder(context) for name in spec.names]
        if len(fns) == 1:
            return ObjectiveFunction(fns[0], spec.names[0]), spec
        names = list(spec.names)
        return CompositeObjectiveFunction(fns, names), spec


objective_factory = ObjectiveFactory()


@objective_factory.register("ce", aliases=("cross_entropy", "loss"))
def _build_ce_objective(context: ObjectiveBuildContext) -> Loss:
    return _wrap_objective(ce_objective, context.ignore_index)


@objective_factory.register(
    "acc_global",
    aliases=("acc", "accuracy", "global_accuracy"),
)
def _build_acc_global_objective(context: ObjectiveBuildContext) -> Loss:
    return _wrap_objective(acc_global_objective, context.ignore_index)


@objective_factory.register(
    "acc_mean",
    aliases=("mean_accuracy", "mean_class_accuracy"),
)
def _build_acc_mean_objective(context: ObjectiveBuildContext) -> Loss:
    return _wrap_objective(acc_mean_objective, context.ignore_index)
