"""No-op attack (identity pass-through)."""

import torch
from torch import Tensor

from .base import (
    AttackTrajectory,
    BaseGradientAttack,
    IterationMetrics,
    StateAdvBest,
    StateBest,
)
from .utils.metrics import attack_success
from .utils.types import Model


class NoAttack(BaseGradientAttack):
    name: str = "no_attack"

    def __init__(
        self,
        model: Model,
        **kwargs,
    ):
        super().__init__(model, **kwargs)
        self.name = "no_attack"

    def perturb(
        self,
        x0: Tensor,
        y0: Tensor | None = None,
        return_trajectory: bool = False,
        **kwargs,
    ) -> Tensor | AttackTrajectory:
        """Run no attack (identity pass-through).

        Args:
            x0: Clean images of shape (B, C, H, W).
            y0: Optional ground truth labels.
            return_trajectory: If True, return full AttackTrajectory with all metrics.

        Returns:
            - Tensor: Clean input (no attack applied).
            - AttackTrajectory: Full trajectory if return_trajectory is True.
        """
        if y0 is None:
            y0 = self.model(x0).argmax(dim=1)

        logits = self.model(x0)
        pred = logits.argmax(1)

        loss_adv = self.loss(logits, y0).flatten(1).mean(1)
        objective_adv = self.objective(logits, y0)
        acc_global, acc_mean = attack_success(
            logits, y0, ignore_index=self.ignore_index
        )

        if not return_trajectory:
            return x0

        state_best = StateBest(
            x_adv=x0,
            g=torch.zeros_like(x0),
            pred_adv=pred,
            loss_adv=loss_adv,
            objective_adv=objective_adv,
            attack_success_global=acc_global,
            attack_success_mean=acc_mean,
        )
        state_adv_best = StateAdvBest(
            x_adv=x0,
            pred_adv=pred,
            attack_success_global=acc_global,
            attack_success_mean=acc_mean,
        )
        intermediate_metrics = {
            0: IterationMetrics(
                iteration=0,
                pred_adv=pred.clone(),
                loss_adv=loss_adv.clone().detach(),
                objective_adv=objective_adv.clone().detach(),
            )
        }
        return AttackTrajectory(
            x_adv=x0,
            state_best=state_best,
            state_adv_best=state_adv_best,
            intermediate_metrics=intermediate_metrics,
        )
