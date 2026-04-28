"""Adversarial attacks for semantic segmentation models."""

from .autopgd import AutoPGDAttack, AutoPGDBase, L2AutoPGDAttack, LinfAutoPGDAttack
from .base import (
    AttackTrajectory,
    BaseGradientAttack,
    IterationMetrics,
    IterativeGradientAttack,
    StateAdvBest,
    StateBest,
)
from .gradient_aggregator import (
    AnnealedCorrectsWeightedSumGradientAggregator,
    BaseWeightedSumGradientAggregator,
    CosineSimilarityWeightedSumGradientAggregator,
    GradientAggregator,
    SumGradientAggregator,
)
from .no_attack import NoAttack
from .norms import L2NormOps, LinfNormOps, NormOps
from .pgd import L2PGDAttack, LinfPGDAttack, PGDAttack
from .utils.losses import LossFactory, LossSpec, loss_factory
from .utils.metrics import ObjectiveFactory, ObjectiveSpec, objective_factory

__all__ = [
    # Attacks (backward-compatible factories)
    "PGDAttack",
    "AutoPGDAttack",
    # Typed attacks
    "L2PGDAttack",
    "LinfPGDAttack",
    "L2AutoPGDAttack",
    "LinfAutoPGDAttack",
    # Base classes
    "BaseGradientAttack",
    "IterativeGradientAttack",
    "AutoPGDBase",
    "NoAttack",
    # State dataclasses
    "StateBest",
    "StateAdvBest",
    "IterationMetrics",
    "AttackTrajectory",
    # Norm operations
    "NormOps",
    "L2NormOps",
    "LinfNormOps",
    # Gradient aggregators
    "GradientAggregator",
    "BaseWeightedSumGradientAggregator",
    "SumGradientAggregator",
    "CosineSimilarityWeightedSumGradientAggregator",
    "AnnealedCorrectsWeightedSumGradientAggregator",
    # Factories
    "LossFactory",
    "LossSpec",
    "loss_factory",
    "ObjectiveFactory",
    "ObjectiveSpec",
    "objective_factory",
]
