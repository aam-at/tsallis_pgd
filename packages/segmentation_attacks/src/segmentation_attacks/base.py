from __future__ import absolute_import, annotations, division, print_function

import warnings
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from logging import Logger
from typing import TYPE_CHECKING, Any, Union

import torch

from .utils.debug import (
    assert_greater,
    assert_greater_equal,
    assert_not_none,
    assert_same_length,
)
from .utils.losses import loss_factory
from .utils.metrics import attack_success, objective_factory
from .utils.tensor_ops import batch_where, tensor_detach
from .utils.types import Loss, Model, Tensor


@dataclass(slots=True)
class StateBest:
    """State tracking for the best adversarial example based on objective value."""

    x_adv: Tensor
    g: Tensor
    pred_adv: Tensor
    loss_adv: Tensor
    objective_adv: Tensor
    attack_success_global: Tensor
    attack_success_mean: Tensor


@dataclass(slots=True)
class StateAdvBest:
    """State tracking for the best adversarial example based on attack success."""

    x_adv: Tensor
    pred_adv: Tensor
    attack_success_global: Tensor
    attack_success_mean: Tensor


@dataclass(slots=True)
class IterationMetrics:
    """Metrics at a specific iteration step."""

    iteration: int
    pred_adv: Tensor
    loss_adv: Tensor
    objective_adv: Tensor


@dataclass(slots=True)
class AttackTrajectory:
    """Complete attack trajectory with final result and intermediate states."""

    x_adv: Tensor
    state_best: StateBest
    state_adv_best: StateAdvBest
    intermediate_metrics: dict[int, IterationMetrics]


if TYPE_CHECKING:
    from .gradient_aggregator.base import GradientAggregator
    from .norms import NormOps


class BaseGradientAttack(ABC):
    """Base class for white-box iterative gradient attacks."""

    model: Model
    iterations: list[int] | int
    epsilon: list[float] | float
    stepsize: list[float] | float
    multi_stage: bool
    loss: Loss
    objective: Loss
    gradient_aggregator: GradientAggregator
    class_weights: list[float] | None
    targeted: bool
    restarts: int
    ignore_index: int = 255
    random_start_: bool
    return_adv_best: bool
    adv_threshold: float
    clip_min: float
    clip_max: float
    max_pixel_value: float = 255.0
    rng: torch.Generator
    verbose: bool
    name: str

    def __init__(
        self,
        model: Model,
        iterations: list[int] | int | None = None,
        epsilon: list[float] | float | None = None,
        stepsize: list[float] | float | None = None,
        loss: str | dict = "ce",
        objective: str | list[str] = "ce",
        gradient_aggregator: Union["GradientAggregator", None] = None,
        class_weights: list[float] | None = None,
        targeted: bool = False,
        restarts: int = 1,
        ignore_index: int = 255,
        random_start: bool = True,
        return_best: bool = True,
        adv_threshold: float = 0.99,
        max_pixel_value: float = 255.0,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
        seed: int | None = None,
        verbose: bool = True,
        name: str | None = None,
        sample_logger: Logger | None = None,
        batch_logger: Logger | None = None,
        compile: bool | str | dict[str, Any] = False,
        **kwargs,
    ):
        assert_not_none(model=model)
        assert_not_none(iterations=iterations)
        assert_greater(0, iterations=iterations)
        assert_not_none(epsilon=epsilon)
        assert_greater(0, epsilon=epsilon)
        assert_not_none(stepsize=stepsize)
        assert_greater(0, stepsize=stepsize)
        assert_not_none(loss=loss)
        assert_not_none(objective=objective)
        assert_greater_equal(1, restarts=restarts)
        assert_not_none(max_pixel_value=max_pixel_value)
        assert_greater(0, max_pixel_value=max_pixel_value)
        if seed is not None:
            assert_greater_equal(0, seed=seed)
        if gradient_aggregator is None:
            raise ValueError(
                "`gradient_aggregator` must be provided for gradient-based attacks."
            )

        self.model = model
        self.iterations = iterations
        self.epsilon = epsilon
        self.stepsize = stepsize
        if isinstance(iterations, Sequence):
            self.multi_stage = True
            assert_same_length(
                iterations=iterations, epsilon=epsilon, stepsize=stepsize
            )
            self.epsilon = [v / max_pixel_value for v in epsilon]
            self.stepsize = [v / max_pixel_value for v in stepsize]
        elif epsilon is not None and stepsize is not None:
            self.multi_stage = False
            self.epsilon = epsilon / max_pixel_value
            self.stepsize = stepsize / max_pixel_value
        else:
            self.multi_stage = False
            self.epsilon = epsilon
            self.stepsize = stepsize
        self.class_weights = class_weights
        self.ignore_index = ignore_index
        self._init_loss_fn(loss)
        self._init_objective_fn(objective)
        self.gradient_aggregator = gradient_aggregator.setup(self)
        self.targeted = targeted
        self.restarts = restarts
        self.random_start_ = random_start
        self.return_adv_best = return_best
        self.adv_threshold = adv_threshold
        self.clip_min = clip_min / max_pixel_value
        self.clip_max = clip_max / max_pixel_value
        self.max_pixel_value = max_pixel_value
        self.rng = torch.Generator()
        if seed is None:
            self.rng.seed()
        else:
            self.rng.manual_seed(seed)
        self.verbose = verbose
        if name is not None:
            self.name = name
        # other attributes
        self.sample_logger = sample_logger
        self.batch_logger = batch_logger
        self.input_id = torch.zeros((1,), dtype=torch.int)
        if compile:
            if isinstance(compile, str):
                self._compile(mode=compile)
            elif isinstance(compile, Mapping):
                self._compile(**compile)
            else:
                self._compile()
        if len(kwargs) > 0:
            warnings.warn("Unused kwargs in BaseGradientAttack: " + str(kwargs))

    def _init_loss_fn(self, loss: str | Mapping[str, Any]):
        """Initialize loss function."""
        self.loss, resolved_loss = loss_factory.build(
            loss,
            ignore_index=self.ignore_index,
            class_weights=self.class_weights,
        )
        self.loss_spec = resolved_loss
        self.loss_name = resolved_loss.name
        self.loss_params: dict[str, Any] = dict(resolved_loss.params)

    def _init_objective_fn(self, objective: str | Sequence[str] | Mapping[str, Any]):
        """Initialize objective function."""
        self.objective, resolved_objective = objective_factory.build(
            objective,
            ignore_index=self.ignore_index,
        )
        self.objective_spec = resolved_objective
        self.objective_name = (
            resolved_objective.names[0]
            if len(resolved_objective.names) == 1
            else list(resolved_objective.names)
        )

    def _compile(self, **kwargs) -> None:
        """Compile attack methods with torch.compile.

        Compiles ``get_outputs_at``, ``get_loss_at``, ``get_grad_at``,
        ``project_gradient``, and ``project_perturbation`` to improve
        attack speed.

        Args:
            **kwargs: Forwarded to ``torch.compile`` (e.g. ``mode``,
                ``fullgraph``, ``dynamic``, ``backend``, ``options``).
                ``"reduce-overhead"`` and ``"max-autotune"`` modes use
                CUDA graphs which are incompatible with
                ``torch.func.grad``; they are remapped to compiled-only
                non-cudagraph variants. If an aggressive Inductor mode
                still hits a backend compiler failure at runtime, the
                method is retried with ``mode="default"`` while keeping
                compiled execution.
        """
        mode = kwargs.get("mode")
        _CUDAGRAPH_MODES = {"reduce-overhead", "max-autotune"}
        if mode in _CUDAGRAPH_MODES:
            fallback = "max-autotune-no-cudagraphs" if "autotune" in mode else None
            warnings.warn(
                f"compile mode {mode!r} uses CUDA graphs which are "
                f"incompatible with torch.func.grad inside get_grad_at. "
                f"Falling back to {fallback!r}.",
                stacklevel=2,
            )
            kwargs["mode"] = fallback
        self.get_outputs_at = torch.compile(self.get_outputs_at, **kwargs)
        self.get_loss_at = torch.compile(self.get_loss_at, **kwargs)
        self.get_grad_at = torch.compile(self.get_grad_at, **kwargs)
        self.project_gradient = torch.compile(self.project_gradient, **kwargs)
        self.project_perturbation = torch.compile(self.project_perturbation, **kwargs)

    def get_outputs_at(self, x: Tensor, **kwargs) -> Tensor:
        return self.model(x)

    def get_loss_at(
        self, x: Tensor, y: Tensor, reduce: bool = True, **kwargs
    ) -> tuple[Tensor, Tensor]:
        logits = self.get_outputs_at(x, **kwargs)
        loss = self.loss(logits=logits, labels=y, **kwargs)
        if reduce:
            loss = loss.flatten(1).mean(1)
        if self.targeted:
            loss = -loss
        return loss, logits

    def get_grad_at(
        self,
        x: Tensor,
        y: Tensor,
        **kwargs,
    ) -> tuple[Tensor, Tensor, Tensor]:
        return self.gradient_aggregator.compute_gradient(x, y, **kwargs)

    @abstractmethod
    def perturb(
        self,
        x0: Tensor,
        y0: Tensor | None = None,
        return_trajectory: bool = False,
        **kwargs,
    ) -> Tensor | AttackTrajectory:
        """Run the attack."""
        ...

    def random_start(self, x0: Tensor, y0: Tensor, **kwargs) -> Tensor:
        """Random starting point for the attack algorithm."""
        return_tensor = torch.zeros_like(x0)
        return return_tensor

    def project_gradient(
        self, x0: Tensor, xcurr: Tensor, g: Tensor, **kwargs
    ) -> Tensor:
        """Project gradient direction on the feasible step set."""
        return_tensor = torch.zeros_like(g)
        return return_tensor

    def project_perturbation(self, x0: Tensor, xcurr: Tensor, **kwargs) -> Tensor:
        """Project adversarial example on the feasible set."""
        return_tensor = torch.zeros_like(xcurr)
        return return_tensor

    def __len__(self):
        return 1

    def __str__(self):
        return self.name


class IterativeGradientAttack(BaseGradientAttack):
    """Base class providing loop control for iterative attacks like PGD."""

    norm_ops: "NormOps"

    def __init__(self, model: Model, norm_ops: "NormOps", **kwargs):
        super().__init__(model, **kwargs)
        self.norm_ops = norm_ops

    def random_start(
        self, x0: Tensor, y0: Tensor, epsilon: Tensor | None = None
    ) -> Tensor:
        """Random starting point for the attack algorithm."""
        if epsilon is None:
            assert not self.multi_stage, "epsilon must be provided for multi-ε attack."
            epsilon = self.epsilon * torch.ones(
                x0.shape[0], dtype=x0.dtype, device=x0.device
            )
        return self.norm_ops.random_perturbation(x0, epsilon)

    def project_gradient(
        self,
        x0: Tensor,
        xcurr: Tensor,
        g: Tensor,
        stepsize: Tensor | None = None,
        **kwargs,
    ) -> Tensor:
        """Project gradient direction on the feasible step set."""
        batch_size = xcurr.size(0)
        if stepsize is None:
            assert not self.multi_stage, "stepsize must be provided for multi-ε attack."
            stepsize = self.stepsize * torch.ones(
                batch_size, dtype=xcurr.dtype, device=xcurr.device
            )
        return self.norm_ops.steepest_descent(
            x0, xcurr, g, stepsize, self.clip_min, self.clip_max
        )

    def project_perturbation(
        self, x0: Tensor, xcurr: Tensor, epsilon: Tensor | None = None, **kwargs
    ) -> Tensor:
        """Project adversarial example on the feasible set."""
        batch_size = xcurr.size(0)
        if epsilon is None:
            assert not self.multi_stage, "epsilon must be provided for multi-ε attack."
            epsilon = self.epsilon * torch.ones(
                batch_size, dtype=xcurr.dtype, device=xcurr.device
            )
        return self.norm_ops.project_perturbation(
            x0, xcurr, epsilon, self.clip_min, self.clip_max
        )

    def _log_progress(self, xcurr: Tensor, y0: Tensor, step: int = 0) -> None:
        """Log attack progress metrics to all configured loggers."""
        if self.verbose is False:
            return

        assert_greater_equal(0, step=step)
        step_value = int(step)
        batch_size = xcurr.shape[0]

        with torch.no_grad():
            logits = self.get_outputs_at(xcurr)
        global_error, mean_error = attack_success(
            logits, y0, ignore_index=self.ignore_index
        )
        global_accuracy = (1 - global_error).detach().cpu()
        mean_accuracy = (1 - mean_error).detach().cpu()

        current_batch = max(int(self.input_id.item()) - 1, 0)
        batch_metrics = {
            "global_accuracy": global_accuracy.mean().item(),
            "mean_accuracy": mean_accuracy.mean().item(),
            "batch_id": current_batch,
        }

        start_index = current_batch * batch_size
        sample_metrics = [
            {
                "input_id": start_index + i,
                "global_accuracy": global_accuracy[i].item(),
                "mean_accuracy": mean_accuracy[i].item(),
            }
            for i in range(batch_size)
        ]

        if self.sample_logger is not None:
            for sample_metric in sample_metrics:
                self.sample_logger.log_metrics(sample_metric, step=step_value)
            self.sample_logger.save()
        if self.batch_logger is not None:
            self.batch_logger.log_metrics(batch_metrics, step=step_value)
            self.batch_logger.save()

    def _update_best_states(
        self,
        state_best: StateBest,
        state_adv_best: StateAdvBest,
        x_adv: Tensor,
        g: Tensor,
        loss_adv: Tensor,
        logits_adv: Tensor,
        y0: Tensor,
        objective_adv: Tensor,
        attack_success_global: Tensor,
        attack_success_mean: Tensor,
    ) -> tuple[StateBest, StateAdvBest]:
        """Helper to update best adversarial states."""
        is_best = objective_adv >= state_best.objective_adv
        is_adv_best = attack_success_global >= state_adv_best.attack_success_global

        state_best_new = batch_where(
            state_best,
            tensor_detach(
                StateBest(
                    x_adv=x_adv,
                    g=g,
                    loss_adv=loss_adv,
                    pred_adv=logits_adv.argmax(dim=1),
                    objective_adv=objective_adv,
                    attack_success_global=attack_success_global,
                    attack_success_mean=attack_success_mean,
                )
            ),
            is_best,
        )
        state_adv_best_new = batch_where(
            state_adv_best,
            tensor_detach(
                StateAdvBest(
                    x_adv=x_adv,
                    pred_adv=logits_adv.argmax(dim=1),
                    attack_success_global=attack_success_global,
                    attack_success_mean=attack_success_mean,
                )
            ),
            is_adv_best,
        )
        return state_best_new, state_adv_best_new

    def perturb_once(
        self,
        x0: Tensor,
        y0: Tensor,
        xstart: Tensor | None = None,
        return_trajectory: bool = False,
        **kwargs,
    ) -> Tensor | AttackTrajectory:
        """Run the attack one time."""
        if not torch.compiler.is_compiling():
            self._log_progress(x0, y0, step=0)
        if self.multi_stage:
            return self.perturb_multi(
                x0, y0, xstart, return_trajectory=return_trajectory, **kwargs
            )
        else:
            return self.perturb_single(
                x0, y0, xstart, return_trajectory=return_trajectory, stage=1, **kwargs
            )

    def perturb_multi(
        self,
        x0: Tensor,
        y0: Tensor,
        xstart: Tensor | None = None,
        return_trajectory: bool = False,
        **kwargs,
    ):
        one = torch.ones(x0.shape[0], dtype=x0.dtype, device=x0.device)
        if xstart is not None:
            x_adv = xstart
        else:
            x_adv = (
                self.random_start(x0, y0, epsilon=self.epsilon[0] * one)
                if self.random_start_
                else x0
            )
        stage = 1
        # Intermediate stages use state_best (return_adv_best=False)
        for iterations, epsilon, stepsize in zip(
            self.iterations[:-1], self.epsilon[:-1], self.stepsize[:-1]
        ):
            x_adv = self.perturb_single(
                x0,
                y0,
                xstart=x_adv,
                iterations=iterations,
                epsilon=epsilon * one,
                stepsize=stepsize * one,
                return_trajectory=False,
                stage=stage,
                return_adv_best=False,
                **kwargs,
            )
            stage += 1
        # Last stage uses state_adv_best (return_adv_best=True)
        return self.perturb_single(
            x0,
            y0,
            xstart=x_adv,
            iterations=self.iterations[-1],
            epsilon=self.epsilon[-1] * one,
            stepsize=self.stepsize[-1] * one,
            return_trajectory=return_trajectory,
            stage=stage,
            return_adv_best=True,
            **kwargs,
        )

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
        """Run the attack one time.

        Args:
            x0: Clean images of shape (B, C, H, W).
            y0: Ground truth labels of shape (B, H, W).
            xstart: Optional starting point for the attack.
            iterations: Number of iterations (overrides self.iterations if provided).
            epsilon: Epsilon budget (overrides self.epsilon if provided).
            stepsize: Step size (overrides self.stepsize if provided).
            return_trajectory: If True, return full AttackTrajectory with intermediate metrics.
            stage: Stage index for logging purposes.
            return_adv_best: If True, return state_adv_best; if False, return state_best.
                If None, uses self.return_adv_best.

        Returns:
            - Tensor: Final adversarial examples (default).
            - AttackTrajectory: Full trajectory if return_trajectory is True.
        """
        if return_adv_best is None:
            return_adv_best = self.return_adv_best
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
            assert not self.multi_stage, "epsilon must be provided for multi-ε attack."
            epsilon = self.epsilon * torch.ones(
                x0.shape[0], dtype=x0.dtype, device=x0.device
            )
        if stepsize is None:
            assert not self.multi_stage, "stepsize must be provided for multi-ε attack."
            stepsize = self.stepsize * torch.ones(
                x0.shape[0], dtype=x0.dtype, device=x0.device
            )

        x_adv = self.project_perturbation(
            x0, x_adv, epsilon=epsilon, stepsize=stepsize, **kwargs
        )

        # compute gradient at the starting point
        g, loss_adv, logits_adv = self.get_grad_at(
            x_adv,
            y0,
            iterations=self.iterations,
            step=torch.tensor(0, dtype=torch.int32, device=x_adv.device),
        )
        objective_adv = self.objective(logits_adv, y0)

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

        for i in range(iterations):
            # project gradient after each step
            r = self.project_gradient(
                x0, x_adv, g, epsilon=epsilon, stepsize=stepsize, **kwargs
            )
            x_adv = x_adv + r
            # project total perturbation after each step
            x_adv = self.project_perturbation(
                x0, x_adv, epsilon=epsilon, stepsize=stepsize, **kwargs
            )
            x_adv = x_adv.detach()

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

    def perturb(
        self,
        x0: Tensor,
        y0: Tensor | None = None,
        return_trajectory: bool = False,
        **kwargs,
    ) -> Tensor | AttackTrajectory:
        """Run the attack with multiple restarts if restarts > 0.

        Args:
            x0: Clean images of shape (B, C, H, W).
            y0: Optional ground truth labels. If None, uses model predictions.
            return_trajectory: If True, return AttackTrajectory with intermediate metrics.
            **kwargs: Additional keyword arguments passed to perturb_once.

        Returns:
            - Tensor: Final adversarial examples (default).
            - AttackTrajectory: Full trajectory if return_trajectory is True.

        Note:
            When using restarts with return_trajectory=True, the returned
            trajectory is assembled per sample from the best restart for that
            sample.
        """
        self.input_id = self.input_id + 1
        if y0 is None:
            y0 = self.get_outputs_at(x0).argmax(dim=1)
        if self.restarts > 1:
            one = torch.ones(x0.shape[0], dtype=x0.dtype, device=x0.device)
            x_adv_list = []
            objective_adv_list = []
            trajectory_list = [] if return_trajectory else None

            # First restart
            result = self.perturb_once(
                x0, y0, xstart=x0, return_trajectory=return_trajectory, **kwargs
            )
            if return_trajectory:
                x_adv_list.append(result.x_adv)
                trajectory_list.append(result)
            else:
                x_adv_list.append(result)
            logits_adv = self.get_outputs_at(x_adv_list[-1])
            objective_adv_list.append(self.objective(logits_adv, y0).detach())

            # Remaining restarts
            for i in range(self.restarts - 1):
                x_start = (
                    self.random_start(x0, y0, epsilon=self.epsilon[0] * one)
                    if self.multi_stage
                    else self.random_start(x0, y0)
                )
                result = self.perturb_once(
                    x0,
                    y0=y0,
                    xstart=x_start,
                    return_trajectory=return_trajectory,
                    **kwargs,
                )
                if return_trajectory:
                    x_adv_list.append(result.x_adv)
                    trajectory_list.append(result)
                else:
                    x_adv_list.append(result)
                logits_adv = self.get_outputs_at(x_adv_list[-1])
                objective_adv_list.append(self.objective(logits_adv, y0).detach())

            # Select the best restart independently for each sample.
            objective_adv_all = torch.stack(
                objective_adv_list,
                dim=1,
            )
            best_indx = objective_adv_all.argmax(dim=1)
            x_adv_all = torch.stack(x_adv_list, dim=1)
            index = best_indx.view(-1, 1, *([1] * (x_adv_all.ndim - 2)))
            index = index.expand(-1, 1, *x_adv_all.shape[2:])
            x_adv_best = torch.gather(x_adv_all, 1, index)

            if return_trajectory and trajectory_list is not None:
                # TODO Implement trajectory merging
                raise NotImplementedError()
            return x_adv_best
        else:
            return self.perturb_once(
                x0, y0, return_trajectory=return_trajectory, **kwargs
            )
