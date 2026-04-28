"""Auto-PGD attack with L2 and Linf variants."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from .base import (
    AttackTrajectory,
    IterationMetrics,
    IterativeGradientAttack,
    StateAdvBest,
    StateBest,
)
from .norms import L2NormOps, LinfNormOps, NormOps
from .utils.debug import assert_not_none, assert_same_length
from .utils.metrics import attack_success
from .utils.tensor_ops import tensor_detach
from .utils.types import Model, Tensor


class AutoPGDBase(IterativeGradientAttack):
    """Auto Projected Gradient Descent attack base.

    Reference: Croce, F. & Hein, M. (2020), Reliable evaluation of adversarial
    robustness with an ensemble of attacks, In , International Conference on
    Machine Learning.
    """

    multi_stage: bool
    alpha: float
    rho: float
    _iterations_ckpt: float = 0.22
    _iterations_min: float = 0.06
    _iterations_decr: float = 0.03

    def __init__(
        self,
        model: Model,
        norm_ops: NormOps,
        alpha: float = 0.75,
        rho: float = 0.75,
        iterations: list[int] | int | None = None,
        **kwargs,
    ):
        assert_not_none(iterations=iterations)
        if isinstance(iterations, Sequence):
            kwargs["stepsize"] = [1.0] * len(iterations)
            self.multi_eps = True
            assert_same_length(iterations=iterations, epsilon=kwargs["epsilon"])
        else:
            kwargs["stepsize"] = 1.0
            self.multi_eps = False
        super().__init__(model, norm_ops=norm_ops, iterations=iterations, **kwargs)
        self.alpha = alpha
        self.rho = rho

    def iterations_ckpt(self, iterations: int | None = None) -> int:
        if iterations is None:
            assert not self.multi_stage, (
                "Iterations must be provided for multi-ε attack."
            )
            iterations = self.iterations
        return max(int(iterations * self._iterations_ckpt), 1)

    def iterations_min(self, iterations: int | None = None) -> int:
        if iterations is None:
            assert not self.multi_stage, (
                "Iterations must be provided for multi-ε attack."
            )
            iterations = self.iterations
        return max(int(iterations * self._iterations_min), 1)

    def iterations_decr(self, iterations: int | None = None) -> int:
        if iterations is None:
            assert not self.multi_stage, (
                "Iterations must be provided for multi-ε attack."
            )
            iterations = self.iterations
        return max(int(iterations * self._iterations_decr), 1)

    def perturb_single(
        self,
        x0: Tensor,
        y0: Tensor,
        xstart: Tensor | None = None,
        iterations: int | None = None,
        epsilon: Tensor | None = None,
        stepsize: Tensor | None = None,
        return_trajectory: bool = False,
        stage: int = -1,
        return_adv_best: bool | None = None,
        **kwargs,
    ) -> Tensor | AttackTrajectory:
        """Run the attack one time using torch.while_loop for efficient compilation.

        Args:
            x0 (Tensor): the original input.
            y0 (Tensor): the input label or attack's target (depends on
            'targeted' bool flag).
            xstart (Tensor): the starting point for the attack.
            return_trajectory: If True, return full AttackTrajectory with all metrics.
            return_adv_best: If True, return state_adv_best; if False, return state_best.
                If None, uses self.return_adv_best.

        Returns:
            Tensor: the adversarial input, or AttackTrajectory if return_trajectory=True.
        """
        if return_adv_best is None:
            return_adv_best = self.return_adv_best
        batch_size = x0.size(0)
        if xstart is not None:
            x_adv = xstart
        else:
            x_adv = (
                self.random_start(x0, y0, epsilon=epsilon) if self.random_start_ else x0
            )
        if iterations is None:
            assert not self.multi_stage, (
                "Iterations must be provided for multi-ε attack."
            )
            iterations = self.iterations
        if epsilon is None:
            epsilon = self.epsilon * torch.ones(batch_size, dtype=x0.dtype).to(
                x0.device
            )
        stepsize = 2.0 * epsilon * torch.ones(batch_size, dtype=x0.dtype).to(x0.device)
        x_adv = self.project_perturbation(
            x0, x_adv, epsilon=epsilon, stepsize=stepsize, **kwargs
        )

        # compute the gradient at the starting point
        g, loss_adv, logits_adv = self.get_grad_at(
            x_adv,
            y0,
            iterations=iterations,
            step=torch.tensor(0, dtype=torch.int32, device=x_adv.device),
        )
        objective_adv = self.objective(logits_adv, y0)
        x_adv_prev = x_adv.clone().detach()

        # Pre-allocate tensors for saved intermediate states
        intermediate_metrics: dict[int, IterationMetrics] = {}
        if return_trajectory:
            intermediate_metrics[0] = IterationMetrics(
                iteration=0,
                pred_adv=logits_adv.argmax(dim=1),
                loss_adv=loss_adv.clone().detach(),
                objective_adv=objective_adv.clone().detach(),
            )

        # statistics to track the best adversarial example
        attack_success_global, attack_success_mean = attack_success(
            logits_adv, y0, ignore_index=self.ignore_index
        )
        state_best = tensor_detach(
            StateBest(
                x_adv=x_adv,
                g=g,
                pred_adv=logits_adv.argmax(dim=1),
                loss_adv=loss_adv,
                objective_adv=objective_adv,
                attack_success_global=attack_success_global,
                attack_success_mean=attack_success_mean,
            ),
        )
        state_adv_best = tensor_detach(
            StateAdvBest(
                x_adv=x_adv,
                pred_adv=logits_adv.argmax(dim=1),
                attack_success_global=attack_success_global,
                attack_success_mean=attack_success_mean,
            )
        )

        # checkpoints counter
        j = 0
        k = self.iterations_ckpt(iterations)
        # statistics to track loss to check oscillation
        objective_adv_steps = torch.empty(
            (iterations, *objective_adv.shape),
            dtype=objective_adv.dtype,
            device=objective_adv.device,
        )
        objective_last_check = loss_adv.clone().detach()
        reduced_last_check = torch.ones_like(loss_adv)

        for i in range(iterations):
            grad_m = x_adv - x_adv_prev
            x_adv_prev = x_adv.clone().detach()
            # update only if the prediction is equal to the original label
            # project gradient after each step
            r = self.project_gradient(x0, x_adv, g, stepsize=stepsize, **kwargs)
            if i > 0:
                # momentum for L2 and Linf norms
                x_adv_1 = self.project_perturbation(
                    x0, x_adv + r, epsilon=epsilon, **kwargs
                )
                r_1 = x_adv_1 - x_adv
                x_adv_1 = x_adv + r_1 * self.alpha + (1 - self.alpha) * grad_m
            else:
                x_adv_1 = x_adv + r
            # project total perturbation after each step
            x_adv_new = self.project_perturbation(
                x0, x_adv_1, epsilon=epsilon, **kwargs
            )
            x_adv = x_adv_new.detach()
            # compute new loss and gradient
            if i == iterations - 1:
                loss_adv, logits_adv = self.get_loss_at(
                    x_adv,
                    y0,
                    iterations=iterations,
                    step=torch.tensor(i + 1, dtype=torch.int32, device=x_adv.device),
                )
            else:
                g, loss_adv, logits_adv = self.get_grad_at(
                    x_adv,
                    y0,
                    iterations=iterations,
                    step=torch.tensor(i + 1, dtype=torch.int32, device=x_adv.device),
                )
                x_adv = x_adv.detach()

            objective_adv = self.objective(logits_adv, y0)
            attack_success_global, attack_success_mean = attack_success(
                logits_adv, y0, ignore_index=self.ignore_index
            )

            # track and update the best adversarial example
            state_best, state_adv_best = self._update_best_states(
                state_best,
                state_adv_best,
                x_adv,
                g,
                loss_adv,
                logits_adv,
                y0,
                objective_adv,
                attack_success_global,
                attack_success_mean,
            )

            # Save intermediate metrics at each iteration
            if return_trajectory:
                iteration_idx = i + 1
                pred_to_save = (
                    state_adv_best.pred_adv if return_adv_best else state_best.pred_adv
                )
                intermediate_metrics[iteration_idx] = IterationMetrics(
                    iteration=iteration_idx,
                    pred_adv=pred_to_save.clone(),
                    loss_adv=loss_adv.clone().detach(),
                    objective_adv=objective_adv.clone().detach(),
                )

            objective_adv_steps[i] = objective_adv.detach()
            # check step size
            j += 1
            if j == k:
                # condition 1: objective oscillation
                fl_oscillation = self.check_oscillation(
                    objective_adv_steps, i, k, threshold=self.rho
                )
                # condition 2: no improvement + not reduced in the last check
                fl_reduce_no_impr = (1.0 - reduced_last_check) * (
                    objective_last_check >= state_best.objective_adv
                )
                fl_should_reduce = torch.max(fl_oscillation, fl_reduce_no_impr)
                # update losses and checks
                reduced_last_check = fl_should_reduce.clone()
                objective_last_check = state_best.objective_adv.clone().detach()
                fl_mask = fl_should_reduce > 0
                fl_mask_exp = fl_mask.view(-1, *([1] * (x_adv.ndim - 1)))
                stepsize = torch.where(fl_mask, stepsize / 2.0, stepsize)
                x_adv = torch.where(fl_mask_exp, state_best.x_adv, x_adv)
                g = torch.where(fl_mask_exp, state_best.g, g)
                k = max(
                    k - self.iterations_decr(iterations),
                    self.iterations_min(iterations),
                )
                j = 0
        x_adv = self.project_perturbation(
            x0,
            state_adv_best.x_adv if return_adv_best else state_best.x_adv,
            epsilon=epsilon,
            stepsize=stepsize,
            **kwargs,
        ).detach()
        if not torch.compiler.is_compiling():
            self._log_progress(x_adv, y0, step=stage)

        # Return based on output mode
        if return_trajectory:
            return AttackTrajectory(
                x_adv=x_adv,
                state_best=state_best,
                state_adv_best=state_adv_best,
                intermediate_metrics=intermediate_metrics,
            )
        return x_adv

    def check_oscillation(self, x, j, k, threshold=0.75):
        """Check for objective oscillation during attack.

        Args:
            x: Tensor of shape [iterations, batch_size] with objective values.
            j: Current iteration index.
            k: Window size for oscillation check.
            threshold: Threshold for oscillation detection.

        Returns:
            Tensor of shape [batch_size] with 1.0 for oscillating samples.
        """
        if k <= 0 or j - k < 0:
            return torch.zeros(x.shape[1], dtype=x.dtype, device=x.device)

        # Use fixed indexing pattern for torch.compile compatibility
        # Slice the last k+1 elements and compare consecutive pairs
        x_curr = x.narrow(0, j - k + 1, k)  # x[j-k+1:j+1]
        x_prev = x.narrow(0, j - k, k)  # x[j-k:j]

        diff = x_curr > x_prev
        t = diff.float().sum(dim=0)
        return (t <= k * threshold).to(dtype=x.dtype)


class L2AutoPGDAttack(AutoPGDBase):
    """L2 Auto-PGD attack."""

    name: str = "autopgd_l2"

    def __init__(self, model: Model, **kwargs):
        super().__init__(model, norm_ops=L2NormOps(), **kwargs)


class LinfAutoPGDAttack(AutoPGDBase):
    """Linf Auto-PGD attack."""

    name: str = "autopgd_linf"

    def __init__(self, model: Model, **kwargs):
        super().__init__(model, norm_ops=LinfNormOps(), **kwargs)


class AutoPGDAttack:
    """Factory that returns L2AutoPGDAttack or LinfAutoPGDAttack based on ``ord``.

    Preserves backward compatibility with ``AutoPGDAttack(model, ord=2, ...)``.
    """

    def __new__(cls, model: Model, ord: str | int = np.inf, **kwargs):
        if ord in (2, "2", "L2"):
            return L2AutoPGDAttack(model, **kwargs)
        elif ord in (np.inf, "Linf"):
            return LinfAutoPGDAttack(model, **kwargs)
        raise ValueError(f"{ord=} is not supported. Use 2, 'L2', np.inf, or 'Linf'.")
