from __future__ import annotations

import glob as _glob
import json
import os
import random
import statistics
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import rootutils
import torch
import torch.utils.data as data
from lightning import LightningDataModule
from omegaconf import DictConfig, ListConfig
from PIL import Image
from rich.console import Console
from tqdm import tqdm

_ = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from data.configs import register_configs  # noqa: E402
from utils.cli import setup_resolvers  # noqa: E402
from utils.pylogger import RankedLogger  # noqa: E402
from utils.runner import setup_extras  # noqa: E402

console = Console()
log = RankedLogger(__name__, rank_zero_only=True)

SEED = 225
random.seed(SEED)
np.random.seed(SEED)

g = torch.Generator()
g.manual_seed(SEED)

_PRED_SUBDIR = "gray"
_PRED_SUFFIX = "_adv.png"
_GLOB_CHARS = set("*?[]")

TaskSpec = str | list[str]


@dataclass(slots=True)
class Subset:
    start: int | None = None
    stop: int | None = None
    step: int | None = None


def _set_global_seed(seed: int) -> None:
    global SEED

    SEED = int(seed)
    random.seed(SEED)
    np.random.seed(SEED)
    g.manual_seed(SEED)


def seed_worker(worker_id):
    # Seed everything
    del worker_id

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def update_fn(data, target, running_stats, stat="intersection", n_cls=21):
    intersection_all1 = data.squeeze() == target.squeeze()

    if stat == "intersection":
        for cl in range(n_cls):
            ind = (target == cl).squeeze()
            running_stats[cl] += intersection_all1[ind].float().sum()
    else:
        for cl in range(n_cls):
            ind = (target == cl).squeeze()
            running_stats[cl] += (
                ind.float().sum()
                + (data.squeeze() == cl).float().sum()
                - intersection_all1[ind].float().sum()
            )

    return running_stats


def update_fn_indiv(data, target, stat="intersection", n_cls=21):
    intersection_all1 = data.squeeze() == target.squeeze()
    indiv_stats = torch.zeros(n_cls)

    if stat == "intersection":
        for cl in range(n_cls):
            ind = (target == cl).squeeze()
            indiv_stats[cl] = intersection_all1[ind].float().sum()
    else:
        for cl in range(n_cls):
            ind = (target == cl).squeeze()
            indiv_stats[cl] = (
                ind.float().sum()
                + (data.squeeze() == cl).float().sum()
                - intersection_all1[ind].float().sum()
            )

    return indiv_stats


def _compute_miou(inters, union):
    iou = []
    for a, b in zip(inters, union):
        if b == 0:  # Skip empty classes.
            continue
        iou.append(a.item() / b.item())

    return statistics.mean(iou)


def _compute_miou_subtraction(running_int, running_union, inters, union):
    iou = []
    uni = []
    miou = []
    ct = 0
    for a, b, c, d in zip(running_int, running_union, inters, union):
        if b == 0:  # Skip empty classes.
            continue
        iou.append(a.item() + c.item())
        uni.append(b.item() + d.item())
        miou.append(iou[ct] / (uni[ct] + 1e-8))
        ct += 1

    return statistics.mean(miou), iou, uni


def _is_glob(pattern: str) -> bool:
    return any(char in pattern for char in _GLOB_CHARS)


def _discover_task_dirs(tasks: TaskSpec) -> list[str]:
    """Resolve task directories from a concrete path or a glob/list of globs."""
    items = [tasks] if isinstance(tasks, str) else list(tasks)
    if not items:
        raise FileNotFoundError("tasks is empty")

    pool: list[str] = []
    for item in items:
        item = os.fspath(item)
        if _is_glob(item):
            matches = sorted(path for path in _glob.glob(item) if os.path.isdir(path))
            if not matches:
                raise FileNotFoundError(f"Glob matched no task directories: {item}")
            pool.extend(matches)
            continue

        if not os.path.isdir(item):
            raise FileNotFoundError(f"Not a directory and not a glob: {item}")
        pool.append(item)

    deduped = sorted(dict.fromkeys(os.path.normpath(path) for path in pool))
    if not deduped:
        raise FileNotFoundError(f"tasks resolved to empty pool: {tasks}")
    return deduped


def _load_pred_png(task_dir: str, image_name: str) -> torch.Tensor:
    """Load ``<task_dir>/val/gray/<image_name>_adv.png`` as a long tensor."""
    path = Path(task_dir) / "val" / _PRED_SUBDIR / f"{image_name}{_PRED_SUFFIX}"
    if not path.is_file():
        raise FileNotFoundError(f"Prediction PNG not found: {path}")
    with Image.open(path) as image:
        arr = np.array(image, dtype=np.int64, copy=True)
    if arr.ndim != 2:
        raise ValueError(
            f"Expected a single-channel PNG at {path}, got shape {arr.shape}"
        )
    return torch.from_numpy(arr)


def _image_name_from_path(image_path: str) -> str:
    return os.path.splitext(os.path.basename(image_path))[0]


def _subset_indices(val_data, subset: Subset | None) -> list[int]:
    n_items = len(val_data)
    if subset is None:
        return list(range(n_items))

    start = 0 if subset.start is None else max(0, int(subset.start))
    stop = n_items if subset.stop is None else min(n_items, int(subset.stop))
    step = 1 if subset.step is None else max(1, int(subset.step))
    if start >= stop:
        raise ValueError(f"Empty subset: start={start} stop={stop} step={step}")
    return list(range(start, stop, step))


def _per_image_int_union(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-class intersection and union for one image."""
    pred = pred.reshape(-1).long()
    target = target.reshape(-1).long()
    valid = (target != ignore_index) & (target >= 0) & (target < num_classes)
    if not valid.any():
        return (
            torch.zeros(num_classes, dtype=torch.float64),
            torch.zeros(num_classes, dtype=torch.float64),
        )

    pred_valid = pred[valid].clamp(0, num_classes - 1)
    target_valid = target[valid]
    indices = target_valid * num_classes + pred_valid
    confusion = torch.bincount(indices, minlength=num_classes * num_classes).reshape(
        num_classes, num_classes
    )
    diagonal = torch.diag(confusion)
    intersection = diagonal.to(torch.float64)
    union = (confusion.sum(0) + confusion.sum(1) - diagonal).to(torch.float64)
    return intersection, union


def _miou_from_iu(intersection: torch.Tensor, union: torch.Tensor) -> float:
    """Compute mIoU over classes with nonzero union."""
    mask = union > 0
    if not mask.any():
        return float("nan")
    return (intersection[mask] / union[mask]).mean().item()


def _json_safe(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _normalize_task_spec(
    tasks: str | ListConfig | list[str] | tuple[str, ...],
) -> TaskSpec:
    if isinstance(tasks, (ListConfig, list, tuple)):
        return [str(task) for task in tasks]
    return str(tasks)


def _resolve_task_spec(tasks: TaskSpec) -> TaskSpec:
    if isinstance(tasks, str):
        return os.path.normpath(hydra.utils.to_absolute_path(tasks))
    return [os.path.normpath(hydra.utils.to_absolute_path(task)) for task in tasks]


class evalSEA:
    """
    A class to do worse-case SEA evaluation

    ...

    Attributes
    ----------
    val_data : Object
        a dataset object for the dataset to be evaluated on
    l_outs : list
        loss wise logits per image
    eps : float
        perturbation strength
    n_cls : int
        number of classes in the dataset
    addendum: str
        to locate the necessary  files on disk
    saveDir: str
        output directory location, where statistics will be saved and read from
    saveDict: dict
        output statistics will be saved in this dict, initialized with loss-wise mIoUs
    modelName: str
        modelname to locate necessary I/O files

    Methods
    -------
    get_loader(bs)
        create dataloader for val_data object
        Parameters
        ----------
        bs : int, batch-size, set to 1 for worse-mIoU

    worst_case_miou()
        Computes point-wise worce mIoU

    worse_case_eval(bs)

        Computes worse aACC across attacks
        Parameters
        ----------

        bs : int, batch-size,
        n_batches : int, number of batches to evaluate for : set to -1 for all

    """

    def __init__(
        self,
        val_data,
        l_outs,
        eps,
        n_cls,
        addendum,
        saveDir,
        saveDict,
        modelName,
    ):
        self.val_data = val_data
        self.l_output = l_outs
        self.eps = eps
        self.addendum = addendum
        self.saveDir = saveDir
        self.saveDict = saveDict
        self.modelName = modelName
        self.n_cls = n_cls
        self.los_pairs = [
            "mask-ce-bal",
            "mask-ce-avg",
            "js-avg",
        ]  # standard-SEA attacks

    def get_loader(self, bs=1):
        val_loader = data.DataLoader(
            self.val_data,
            batch_size=bs,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            sampler=None,
            worker_init_fn=seed_worker,
            generator=g,
        )
        return val_loader

    def _default_ignore_index(self) -> int:
        for obj in (
            self.val_data,
            getattr(self.val_data, "transforms", None),
            getattr(self.val_data, "info", None),
            getattr(getattr(self.val_data, "transforms", None), "info", None),
        ):
            if obj is not None and getattr(obj, "ignore_index", None) is not None:
                return int(obj.ignore_index)
        raise ValueError(
            "Could not infer ignore_index from val_data; pass ignore_index explicitly."
        )

    def aggregate_worst_acc(
        self,
        tasks: TaskSpec,
        ignore_index: int | None = None,
        subset: Subset | None = None,
    ) -> dict[str, Any]:
        """Worst per-image global pixel accuracy across attacks in ``tasks``."""
        task_dirs = _discover_task_dirs(tasks)
        indices = _subset_indices(self.val_data, subset)
        resolved_ignore_index = (
            int(ignore_index)
            if ignore_index is not None
            else self._default_ignore_index()
        )
        acc_matrix = torch.full(
            (len(task_dirs), len(indices)),
            float("nan"),
            dtype=torch.float64,
        )

        for column, dataset_idx in enumerate(tqdm(indices, desc="Worst-Acc")):
            image_name = _image_name_from_path(self.val_data.images[dataset_idx])
            _, target = self.val_data[dataset_idx]
            target = torch.as_tensor(target).long().cpu()
            valid = (
                (target != resolved_ignore_index)
                & (target >= 0)
                & (target < self.n_cls)
            )
            n_valid = int(valid.sum().item())
            if n_valid == 0:
                continue

            for attack_idx, task_dir in enumerate(task_dirs):
                pred = _load_pred_png(task_dir, image_name)
                if pred.shape != target.shape:
                    raise AssertionError(
                        f"Prediction shape mismatch for {image_name} in {task_dir}: "
                        f"pred={tuple(pred.shape)} target={tuple(target.shape)}"
                    )
                correct = ((pred == target) & valid).sum()
                acc_matrix[attack_idx, column] = correct.to(torch.float64) / n_valid

        valid_columns = torch.isfinite(acc_matrix).all(dim=0)
        if not valid_columns.any():
            raise ValueError("No valid pixels found in the requested subset.")

        worst_per_image = acc_matrix[:, valid_columns].min(dim=0).values
        worst_acc = worst_per_image.mean().item()
        evaluated_indices = [
            indices[position]
            for position in torch.nonzero(valid_columns, as_tuple=False)
            .flatten()
            .tolist()
        ]

        self.saveDict["worst_Acc"] = worst_acc
        # NB: the legacy worse_case_eval stores per-*attack* mean accuracy
        # under "worst_Acc_indiv" (shape [n_attacks]).  The new aggregate
        # stores per-*image* worst accuracy (shape [n_valid_images]) under a
        # different key to avoid silent semantic collision.
        self.saveDict["worst_Acc_per_image"] = worst_per_image
        self.saveDict["worst_Acc_per_attack"] = acc_matrix[:, valid_columns].mean(dim=1)

        return {
            "worst_acc": worst_acc,
            "worst_acc_per_image": worst_per_image,
            "worst_acc_per_attack": acc_matrix[:, valid_columns].mean(dim=1),
            "task_dirs": task_dirs,
            "indices": indices,
            "evaluated_indices": evaluated_indices,
            "ignore_index": resolved_ignore_index,
        }

    def aggregate_worst_miou(
        self,
        tasks: TaskSpec,
        ignore_index: int | None = None,
        subset: Subset | None = None,
        n_rounds: int = 1000,
        tol: float = 1e-6,
    ) -> dict[str, Any]:
        """Worst mIoU across attacks in ``tasks`` via greedy image selection."""
        task_dirs = _discover_task_dirs(tasks)
        indices = _subset_indices(self.val_data, subset)
        resolved_ignore_index = (
            int(ignore_index)
            if ignore_index is not None
            else self._default_ignore_index()
        )
        n_attacks = len(task_dirs)
        n_images = len(indices)
        cons_ints = torch.zeros((n_attacks, n_images, self.n_cls), dtype=torch.float64)
        cons_unions = torch.zeros_like(cons_ints)

        for image_offset, dataset_idx in enumerate(tqdm(indices, desc="Worst-mIoU")):
            image_name = _image_name_from_path(self.val_data.images[dataset_idx])
            _, target = self.val_data[dataset_idx]
            target = torch.as_tensor(target).long().cpu()
            for attack_idx, task_dir in enumerate(task_dirs):
                pred = _load_pred_png(task_dir, image_name)
                if pred.shape != target.shape:
                    raise AssertionError(
                        f"Prediction shape mismatch for {image_name} in {task_dir}: "
                        f"pred={tuple(pred.shape)} target={tuple(target.shape)}"
                    )
                (
                    cons_ints[attack_idx, image_offset],
                    cons_unions[attack_idx, image_offset],
                ) = _per_image_int_union(
                    pred=pred,
                    target=target,
                    num_classes=self.n_cls,
                    ignore_index=resolved_ignore_index,
                )

        selected_attk_indx = [0] * n_images
        running_intersection = cons_ints[0].sum(dim=0).clone()
        running_union = cons_unions[0].sum(dim=0).clone()
        final_miou = _miou_from_iu(running_intersection, running_union)
        console.print(
            f"Miou after {Path(task_dirs[0]).name} > "
            f"[light_steel_blue1]{final_miou}[/light_steel_blue1]"
        )
        console.print(f"[cyan]Starting {n_rounds} random rounds now [/cyan]")

        for _ in range(int(n_rounds)):
            prev_best = final_miou
            shuffled_list = list(range(n_images))
            random.shuffle(shuffled_list)

            for idx in shuffled_list:
                current_attack = selected_attk_indx[idx]
                for attack in range(n_attacks):
                    update_int = (
                        cons_ints[attack, idx, :] - cons_ints[current_attack, idx, :]
                    )
                    update_union = (
                        cons_unions[attack, idx, :]
                        - cons_unions[current_attack, idx, :]
                    )
                    candidate_intersection = running_intersection + update_int
                    candidate_union = running_union + update_union
                    est_miou = _miou_from_iu(candidate_intersection, candidate_union)

                    if est_miou < final_miou:
                        selected_attk_indx[idx] = attack
                        running_intersection = candidate_intersection
                        running_union = candidate_union
                        final_miou = est_miou
                        current_attack = attack

            if prev_best - final_miou <= float(tol):
                break

        self.saveDict["seed"] = SEED
        self.saveDict["final_miou"] = final_miou

        return {
            "final_miou": final_miou,
            "seed": SEED,
            "task_dirs": task_dirs,
            "selected": selected_attk_indx,
            "indices": indices,
            "ignore_index": resolved_ignore_index,
        }

    def worst_case_miou(self):
        running_intersection = [0 for _ in range(self.n_cls)]
        running_union = [0 for _ in range(self.n_cls)]
        selected_attk_indx = []
        data_loader = self.get_loader(bs=1)  # bs = 1 since we do it image-wise

        if not self.l_output:
            self.l_output = []
            for loss_ in self.los_pairs:
                self.l_output.append(
                    torch.load(
                        self.saveDir
                        + f"/argmax-logs/{self.modelName}_{loss_}_{self.eps}.pt"
                    )
                )

        if not os.path.isfile(
            self.saveDir + f"/running_stats/stats_{self.addendum}_{self.eps}.pt"
        ):
            class_wise_logits = torch.stack(self.l_output, dim=0)
            cons_ints = torch.zeros(
                (class_wise_logits.shape[0], class_wise_logits.shape[1], self.n_cls)
            )
            cons_unions = torch.zeros(
                (class_wise_logits.shape[0], class_wise_logits.shape[1], self.n_cls)
            )
            total_points = class_wise_logits.size(1)
            console.print(
                "[light_steel_blue1] Starting worse-mIoU computation, point-wise "
                "[/light_steel_blue1]"
            )

            for i, vals in tqdm(enumerate(data_loader), desc="Worse-mIoU"):
                _, target = vals[0], vals[1]

                pred = class_wise_logits[:, i, :, :]

                attack_to_select = None

                for attack in range(class_wise_logits.size(0)):
                    cons_ints[attack, i, :] = update_fn_indiv(
                        pred[attack].clone(),
                        target,
                        "intersection",
                        self.n_cls,
                    )
                    cons_unions[attack, i, :] = update_fn_indiv(
                        pred[attack].clone(), target, "union", self.n_cls
                    )

                    attack_to_select = 0
                selected_attk_indx.append(attack_to_select)

                running_intersection = update_fn(
                    pred[0, :, :],
                    target,
                    running_intersection,
                    "intersection",
                    self.n_cls,
                )
                running_union = update_fn(
                    pred[0, ::], target, running_union, "union", self.n_cls
                )
                if i + 1 >= total_points:
                    break

            savedict = {
                "run_int_imwise": cons_ints,
                "run_union_imwise": cons_unions,
                "run_intersect_abs": running_intersection,
                "run_union_abs": running_union,
            }
            del class_wise_logits

            torch.save(
                savedict,
                self.saveDir + f"/test_results/stats_{self.addendum}_{self.eps}.pt",
            )

        else:
            tenss = torch.load(
                self.saveDir + f"/test_results/stats_{self.addendum}_{self.eps}.pt"
            )
            cons_ints, cons_unions, running_intersection, running_union = (
                tenss["run_int_imwise"],
                tenss["run_union_imwise"],
                tenss["run_intersect_abs"],
                tenss["run_union_abs"],
            )

        final_miou = _compute_miou(running_intersection, running_union)
        console.print(
            f"Miou after mask-ce-bal > [light_steel_blue1]{final_miou}[/light_steel_blue1]"
        )
        n_rounds = 1000
        selected_attk_indx = [0] * cons_ints.shape[1]
        console.print(
            f"[cyan]Starting {n_rounds} random rounds now [/cyan]",
        )
        prev_best = 10
        for rounds in range(n_rounds):
            del rounds
            shuffled_list = list(range(0, cons_ints.shape[1]))
            random.shuffle(shuffled_list)
            for i, idx in enumerate(shuffled_list):
                del i
                for attack in range(cons_ints.size(0)):
                    est_int = running_intersection.copy()
                    est_union = running_union.copy()

                    update_int = (
                        cons_ints[attack, idx, :]
                        - cons_ints[selected_attk_indx[idx], idx, :]
                    )
                    update_union = (
                        cons_unions[attack, idx, :]
                        - cons_unions[selected_attk_indx[idx], idx, :]
                    )

                    est_miou, new_ints, new_unis = _compute_miou_subtraction(
                        torch.tensor(est_int),
                        torch.tensor(est_union),
                        torch.tensor(update_int),
                        torch.tensor(update_union),
                    )

                    if est_miou < final_miou:
                        selected_attk_indx[idx] = attack
                        running_intersection = new_ints
                        running_union = new_unis
                final_miou = _compute_miou(
                    torch.tensor(running_intersection),
                    torch.tensor(running_union),
                )

            if prev_best - final_miou <= 1e-6:
                break
            else:
                prev_best = final_miou
            final_miou = _compute_miou(
                torch.tensor(running_intersection), torch.tensor(running_union)
            )

        self.saveDict["seed"] = SEED
        self.saveDict["final_miou"] = final_miou

        print("SEA Evaluation complete, saved-dict:")
        print(self.saveDict)

        del cons_ints
        del cons_unions
        del running_intersection
        del running_union
        del data_loader

    def worse_case_eval(self, bs=16, n_batches=-1):
        """Compute worse case aACC across 3-losses for SEA."""

        if not self.l_output:
            self.l_output = []
            for loss_ in self.los_pairs:
                self.l_output.append(
                    torch.load(
                        self.saveDir
                        + f"/argmax-logs/{self.modelName}_{loss_}_{self.eps}.pt"
                    )
                )

        final_acc_1 = None

        class_wise_logits = torch.stack(self.l_output)

        aaacc = []

        data_loader = self.get_loader(bs=bs)

        for i, vals in enumerate(data_loader):
            _, target = vals[0], vals[1]
            batch_size = target.shape[0]
            acc_cls = torch.zeros(class_wise_logits.shape[0], batch_size, self.n_cls)
            n_pxl_cls = torch.zeros(class_wise_logits.shape[0], batch_size, self.n_cls)

            pred = class_wise_logits[:, i * batch_size : i * batch_size + batch_size]

            acc_curr = pred == target

            for cl in range(self.n_cls):
                ind = target == cl
                ind = ind.expand(class_wise_logits.shape[0], -1, -1, -1)
                acc_curr_ = acc_curr * ind
                acc_cls[:, :, cl] += (
                    acc_curr_.view(acc_curr.shape[0], acc_curr.shape[1], -1)
                    .float()
                    .sum(2)
                )
                n_pxl_cls[:, :, cl] += (
                    ind.view(ind.shape[0], ind.shape[1], -1).float().sum(2)
                )

            tenss = acc_cls.sum(2) / n_pxl_cls.sum(2)
            aaacc.append(tenss)

            if i + 1 == n_batches:
                print("enough batches seen")
                break

        final_acc_1 = torch.cat(aaacc, dim=-1)

        worse_1 = final_acc_1.min(0)[0].mean()
        at_w_sum1 = final_acc_1.mean(-1)

        print("SEA evaluated Acc", worse_1)

        self.saveDict["worst_Acc"] = worse_1.item()
        self.saveDict["worst_Acc_indiv"] = at_w_sum1

        del final_acc_1
        del class_wise_logits
        del acc_cls
        del n_pxl_cls
        del data_loader


@hydra.main(version_base="1.3", config_path="../configs", config_name="worse.yaml")
def main(cfg: DictConfig) -> None:
    if torch.cuda.is_available():
        import torch.backends.cudnn as cudnn

        cudnn.benchmark = True
        cudnn.deterministic = False
        torch.set_float32_matmul_precision("high")

    setup_extras(cfg)

    seed = cfg.get("seed")
    if seed is not None:
        _set_global_seed(int(seed))

    if cfg.tasks is None:
        raise ValueError("cfg.tasks must be set to a directory, glob, or list of them.")

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    datamodule.setup()

    split_name = str(cfg.data_split)
    val_data = getattr(datamodule, f"data_{split_name}")
    if val_data is None:
        raise ValueError(f"Datamodule did not populate data_{split_name}.")
    if hasattr(val_data, "sort"):
        val_data.sort()

    tasks = _resolve_task_spec(_normalize_task_spec(cfg.tasks))

    subset = None
    if cfg.get("subset") is not None and any(
        cfg.subset.get(key) is not None for key in ("start", "stop", "step")
    ):
        subset = Subset(
            start=cfg.subset.get("start"),
            stop=cfg.subset.get("stop"),
            step=cfg.subset.get("step"),
        )

    attack_eval = evalSEA(
        val_data=val_data,
        l_outs=None,
        eps=0.0,
        n_cls=int(datamodule.num_classes),
        addendum=split_name,
        saveDir=str(cfg.paths.output_dir),
        saveDict={},
        modelName=str(cfg.task_name),
    )

    ignore_index = cfg.get("ignore_index")
    if ignore_index is None:
        data_info = cfg.data.get("info") if cfg.get("data") is not None else None
        if data_info is not None and data_info.get("ignore_index") is not None:
            ignore_index = int(data_info.ignore_index)
        else:
            ignore_index = attack_eval._default_ignore_index()
    else:
        ignore_index = int(ignore_index)

    normalized_metrics = _normalize_task_spec(cfg.metrics)
    metrics_requested = (
        [normalized_metrics]
        if isinstance(normalized_metrics, str)
        else [str(metric) for metric in normalized_metrics]
    )
    results: dict[str, Any] = {
        "cfg_tasks": tasks,
        "subset": None if subset is None else asdict(subset),
        "ignore_index": ignore_index,
    }

    if "worst_acc" in metrics_requested:
        log.info("Computing worst pixel accuracy from saved PNG predictions.")
        results["worst_acc"] = attack_eval.aggregate_worst_acc(
            tasks=tasks,
            ignore_index=ignore_index,
            subset=subset,
        )
    if "worst_miou" in metrics_requested:
        log.info("Computing worst mIoU from saved PNG predictions.")
        results["worst_miou"] = attack_eval.aggregate_worst_miou(
            tasks=tasks,
            ignore_index=ignore_index,
            subset=subset,
            n_rounds=int(cfg.n_rounds),
            tol=float(cfg.tol),
        )

    output_path = Path(str(cfg.output_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(results), handle, indent=2)
    log.info(f"Worse-case results written to {output_path}")


if __name__ == "__main__":
    register_configs()
    setup_resolvers()
    main()
