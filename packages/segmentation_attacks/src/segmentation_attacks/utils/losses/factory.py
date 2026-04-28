"""Registry-backed LossFactory and built-in loss registrations."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

from ..types import Loss
from .common import _wrap_loss
from .divergence import dice_loss, js_loss
from .margin import cw_loss, dlr_loss
from .tsallis import adaptive_tsallis_ce_loss, tsallis_ce_loss, tsallis_js_loss


@dataclass(frozen=True)
class LossBuildContext:
    """Inputs required to construct a reusable loss callable."""

    ignore_index: int
    params: Mapping[str, Any]
    class_weights: Sequence[float] | None = None


LossBuilder = Callable[[LossBuildContext], Loss]


@dataclass(frozen=True)
class LossSpec:
    """Normalized loss configuration used by attacks."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LossRegistration:
    """Registry entry describing a canonical loss and its aliases."""

    name: str
    builder: LossBuilder
    aliases: tuple[str, ...] = ()


class LossFactory:
    """Registry-backed factory for resolving and building attack losses."""

    def __init__(self) -> None:
        self._entries: dict[str, LossRegistration] = {}
        self._aliases: dict[str, str] = {}

    def register(
        self, name: str, *, aliases: Sequence[str] = ()
    ) -> Callable[[LossBuilder], LossBuilder]:
        canonical = name.lower()
        normalized_aliases = tuple(alias.lower() for alias in aliases)

        def decorator(builder: LossBuilder) -> LossBuilder:
            self._register(
                LossRegistration(
                    name=canonical,
                    builder=builder,
                    aliases=normalized_aliases,
                )
            )
            return builder

        return decorator

    def _register(self, registration: LossRegistration) -> None:
        if registration.name in self._entries or registration.name in self._aliases:
            raise ValueError(f"Loss '{registration.name}' is already registered")
        for alias in registration.aliases:
            if alias == registration.name:
                continue
            if alias in self._entries or alias in self._aliases:
                raise ValueError(
                    f"Alias '{alias}' for loss '{registration.name}' is already registered"
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

    def normalize_name(self, loss_name: str) -> str:
        lowered = loss_name.lower()
        if lowered in self._entries:
            return lowered
        alias = self._aliases.get(lowered)
        if alias is not None:
            return alias
        raise ValueError(
            f"Unknown loss '{loss_name}'. Supported names: {sorted(self.supported_names())}"
        )

    def resolve(self, loss: str | Mapping[str, Any] | LossSpec) -> LossSpec:
        if isinstance(loss, LossSpec):
            return LossSpec(
                name=self.normalize_name(loss.name),
                params=dict(loss.params),
            )
        if isinstance(loss, str):
            return LossSpec(name=self.normalize_name(loss))
        if not isinstance(loss, Mapping):
            raise TypeError(
                f"loss spec must be str | Mapping, got {type(loss).__name__}"
            )
        loss_type = loss.get("type")
        if not isinstance(loss_type, str):
            raise TypeError("loss mapping must contain a string 'type' key")
        params = loss.get("params", {})
        if params is None:
            params = {}
        if not isinstance(params, Mapping):
            raise TypeError(
                f"loss['params'] must be a mapping, got {type(params).__name__}"
            )
        return LossSpec(name=self.normalize_name(loss_type), params=dict(params))

    def build(
        self,
        loss: str | Mapping[str, Any] | LossSpec,
        *,
        ignore_index: int,
        class_weights: Sequence[float] | None = None,
        extra_params: Mapping[str, Any] | None = None,
    ) -> tuple[Loss, LossSpec]:
        spec = self.resolve(loss)
        params = dict(spec.params)
        if extra_params is not None:
            params.update(dict(extra_params))
        registration = self._entries[spec.name]
        context = LossBuildContext(
            ignore_index=ignore_index,
            params=params,
            class_weights=class_weights,
        )
        return registration.builder(context), LossSpec(name=spec.name, params=params)

    @property
    def canonical_names(self) -> tuple[str, ...]:
        return tuple(self._entries)

    @property
    def aliases(self) -> dict[str, str]:
        return dict(self._aliases)


# ---------------------------------------------------------------------------
# Helpers for builder functions
# ---------------------------------------------------------------------------


def _require_param(params: Mapping[str, Any], key: str) -> Any:
    if key not in params:
        raise ValueError(f"Missing loss parameter '{key}'")
    return params[key]


def _require_class_weights(
    class_weights: Sequence[float] | None,
) -> Sequence[float]:
    if class_weights is None:
        raise ValueError("class_weights must be provided for balanced losses")
    return class_weights


# ---------------------------------------------------------------------------
# Singleton factory with built-in registrations
# ---------------------------------------------------------------------------

loss_factory = LossFactory()


@loss_factory.register("ce", aliases=("cross_entropy", "cross-entropy"))
def _build_ce(context: LossBuildContext) -> Loss:
    return _wrap_loss(F.cross_entropy, context.ignore_index)


@loss_factory.register("bal_ce")
def _build_balanced_ce(context: LossBuildContext) -> Loss:
    weights = _require_class_weights(context.class_weights)

    def loss_fn(logits, labels, **kw):
        w = torch.as_tensor(weights, device=logits.device)
        return F.cross_entropy(
            logits,
            labels,
            ignore_index=context.ignore_index,
            weight=w,
            reduction="none",
        )

    return loss_fn


@loss_factory.register("masked_ce")
def _build_masked_ce(context: LossBuildContext) -> Loss:
    def loss_fn(logits, labels, **kw):
        corrects_mask = logits.max(1)[1] == labels
        loss = F.cross_entropy(
            logits, labels, ignore_index=context.ignore_index, reduction="none"
        )
        return corrects_mask.float().detach() * loss

    return loss_fn


@loss_factory.register("bal_masked_ce")
def _build_balanced_masked_ce(context: LossBuildContext) -> Loss:
    weights = _require_class_weights(context.class_weights)

    def loss_fn(logits, labels, **kw):
        corrects_mask = logits.max(1)[1] == labels
        w = torch.as_tensor(weights, device=logits.device)
        loss = F.cross_entropy(
            logits,
            labels,
            ignore_index=context.ignore_index,
            weight=w,
            reduction="none",
        )
        return corrects_mask.float().detach() * loss

    return loss_fn


@loss_factory.register("cw", aliases=("margin",))
def _build_cw(context: LossBuildContext) -> Loss:
    return _wrap_loss(cw_loss, context.ignore_index)


@loss_factory.register("dlr")
def _build_dlr(context: LossBuildContext) -> Loss:
    return _wrap_loss(dlr_loss, context.ignore_index)


@loss_factory.register("js")
def _build_js(context: LossBuildContext) -> Loss:
    return _wrap_loss(js_loss, context.ignore_index)


@loss_factory.register("dice")
def _build_dice(context: LossBuildContext) -> Loss:
    return _wrap_loss(dice_loss, context.ignore_index)


@loss_factory.register("tsallis_ce")
def _build_tsallis_ce(context: LossBuildContext) -> Loss:
    default_q = float(_require_param(context.params, "q"))

    def loss_fn(logits, labels, q: float | None = None, **kw):
        return tsallis_ce_loss(
            logits,
            labels,
            q_param=q if q is not None else default_q,
            ignore_index=context.ignore_index,
            reduction="none",
        )

    return loss_fn


@loss_factory.register("adaptive_tsallis_ce")
def _build_adaptive_tsallis_ce(context: LossBuildContext) -> Loss:
    def loss_fn(
        logits,
        labels,
        step: int = -1,
        iterations: int = -1,
        q_param: torch.Tensor | None = None,
    ):
        if not torch.compiler.is_compiling():
            assert iterations > 0, f"iterations must be positive, got {iterations}"
            assert step >= 0, f"step must be non-negative, got {step}"
        return adaptive_tsallis_ce_loss(
            logits,
            labels,
            params=context.params,
            step=step,
            iterations=iterations,
            ignore_index=context.ignore_index,
            reduction="none",
        )

    return loss_fn


@loss_factory.register("tsallis_js")
def _build_tsallis_js(context: LossBuildContext) -> Loss:
    q_param = float(_require_param(context.params, "q"))
    return _wrap_loss(tsallis_js_loss, context.ignore_index, q_param=q_param)
