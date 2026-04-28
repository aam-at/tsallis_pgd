import os
from collections import OrderedDict
from itertools import islice
from typing import Any, Dict, List, Optional, Tuple

import cv2
import hydra
import lightning as L
import numpy as np
import rootutils
import torch
from lightning import LightningDataModule, LightningModule
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

_ = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from data.configs import register_configs  # noqa: E402
from data.utils import (  # noqa: E402
    DataSplit,
    colorize,
    validate_enum_value,
)
from segmentation_attacks.base import BaseGradientAttack  # noqa: E402
from utils.cli import setup_resolvers  # noqa: E402
from utils.instantiators import instantiate_loggers  # noqa: E402
from utils.logging import log_hyperparameters  # noqa: E402
from utils.metrics import MeanMetricsDictionary  # noqa: E402
from utils.pylogger import RankedLogger  # noqa: E402
from utils.rich import AttackProgress  # noqa: E402
from utils.runner import (  # noqa: E402
    check_existing_run_dir,
    check_makedirs,
    setup_extras,
    task_wrapper,
)
from utils.segmentation import ConfusionMatrix  # noqa: E402

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def run_attack(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best
    weights obtained during training.

    This method is wrapped in optional @task_wrapper decorator, that
    controls the behavior during failure. Useful for multiruns, saving
    info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated
        objects.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    datamodule.setup()

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    model.eval()

    log.info("Instantiating loggers...")
    loggers: List[Logger] = instantiate_loggers(cfg.get("loggers"))

    log.info(f"Instantiating attack <{cfg.attack._target_}>")
    attack: BaseGradientAttack = hydra.utils.instantiate(
        cfg.attack, model=model, class_weights=datamodule.get_weights()
    )
    object_dict = {
        "cfg": cfg,
        "data": datamodule,
        "model": model,
        "attack": attack,
    }

    if loggers:
        log.info("Logging hyperparameters!")
        log_hyperparameters(loggers, object_dict)

    metrics = ["AllAcc", "mAcc", "mIoU"]
    categories = ["Original", "Adversarial"]
    metrics_dict = MeanMetricsDictionary()
    confusion_matrix_orig = ConfusionMatrix(num_classes=datamodule.info.num_classes)
    confusion_matrix_adv = ConfusionMatrix(num_classes=datamodule.info.num_classes)
    confusion_matrix_at_iter = {}
    for i in cfg.return_at_iterations:
        confusion_matrix_at_iter[i] = ConfusionMatrix(
            num_classes=datamodule.info.num_classes
        )
        categories.append(f"Adversarial_{i}")

    # get dataset
    dataset = getattr(datamodule, f"data_{cfg.data_split}")
    dataset.sort()
    class_names = dataset.get_class_names()
    colormap = dataset.get_palette()
    i = 0
    try:
        image_list = dataset.images
        dataloader = getattr(datamodule, f"{cfg.data_split}_dataloader")()
        max_batches = cfg.get("total_batches")
        batches_cap = (
            int(max_batches) if max_batches is not None and max_batches > 0 else None
        )
        total_batches = (
            len(dataloader)
            if batches_cap is None
            else min(batches_cap, len(dataloader))
        )
        dataloader_iter = (
            enumerate(islice(dataloader, batches_cap), start=0)
            if batches_cap is not None
            else enumerate(dataloader, start=0)
        )
        # from contextlib import nullcontext
        # with nullcontext():
        with AttackProgress(
            metrics=metrics, categories=categories, auto_refresh=False
        ) as progress:
            attack_task = progress.add_task("Running attack", total=total_batches)
            batch_size = (
                cfg.data.batch_size
                if cfg.data_split == DataSplit.TRAIN
                else cfg.data.test_batch_size
            )
            for i, (images, labels) in dataloader_iter:
                images = images.to(model.device)
                labels = labels.long().to(model.device)
                valid_indices = labels != datamodule.info.ignore_index
                # clean accuracy
                outs_orig = model(images).detach()
                pred_orig = outs_orig.argmax(dim=1)
                pred_orig[~valid_indices] = datamodule.info.ignore_index
                confusion_matrix_orig.update(labels, pred_orig)
                # metrics_dict["acc_orig"].update(acc_orig.cpu())
                return_trajectory = len(cfg.return_at_iterations) > 0
                attack_results = attack.perturb(
                    images,
                    labels,
                    return_trajectory=return_trajectory,
                )
                if return_trajectory:
                    images_adv = attack_results.x_adv
                    state_at_iters = attack_results.intermediate_metrics
                else:
                    images_adv = attack_results
                    state_at_iters = {}
                outs_adv = model(images_adv).detach()

                pred_adv = outs_adv.argmax(dim=1)
                pred_adv[~valid_indices] = datamodule.info.ignore_index
                confusion_matrix_adv.update(labels, pred_adv)
                # metrics_dict["acc_adv"].update(acc_adv.cpu())
                # update metrics
                values = {}
                acc_global_orig, accs_orig, ious_orig = confusion_matrix_orig.compute()
                acc_global_adv, accs_adv, ious_adv = confusion_matrix_adv.compute()
                metrics_dict["acc_orig"] = acc_global_orig.item()
                metrics_dict["miou_orig"] = ious_orig.nanmean().item()
                metrics_dict["macc_orig"] = accs_orig.nanmean().item()
                metrics_dict["acc_adv"] = acc_global_adv.item()
                metrics_dict["miou_adv"] = ious_adv.nanmean().item()
                metrics_dict["macc_adv"] = accs_adv.nanmean().item()
                # update values for progress bar status
                values["AllAcc"] = {
                    "Original": f"{metrics_dict['acc_orig']:.4f}",
                    "Adversarial": f"{metrics_dict['acc_adv']:.4f}",
                }
                values["mIoU"] = {
                    "Original": f"{metrics_dict['miou_orig']:.4f}",
                    "Adversarial": f"{metrics_dict['miou_adv']:.4f}",
                }
                values["mAcc"] = {
                    "Original": f"{metrics_dict['macc_orig']:.4f}",
                    "Adversarial": f"{metrics_dict['macc_adv']:.4f}",
                }

                # adversarial accuracy at iterations
                for iteration in cfg.return_at_iterations:
                    pred_adv_i = state_at_iters[iteration].pred_adv
                    pred_adv_i[labels == datamodule.info.ignore_index] = (
                        datamodule.info.ignore_index
                    )
                    confusion_matrix_at_iter[iteration].update(labels, pred_adv_i)
                    acc_global_adv_i, accs_adv_i, ious_adv_i = confusion_matrix_at_iter[
                        iteration
                    ].compute()
                    # metrics_dict[f"acc_adv_{iteration}"].update(acc_adv_iter.cpu())
                    metrics_dict[f"acc_adv_{iteration}"] = acc_global_adv_i.item()
                    metrics_dict[f"miou_adv_{iteration}"] = ious_adv_i.nanmean().item()
                    metrics_dict[f"macc_adv_{iteration}"] = accs_adv_i.nanmean().item()
                    # update values for progress bar status
                    values["AllAcc"][f"Adversarial_{iteration}"] = (
                        f"{metrics_dict[f'acc_adv_{iteration}']:.4f}"
                    )
                    values["mIoU"][f"Adversarial_{iteration}"] = (
                        f"{metrics_dict[f'miou_adv_{iteration}']:.4f}"
                    )
                    values["mAcc"][f"Adversarial_{iteration}"] = (
                        f"{metrics_dict[f'macc_adv_{iteration}']:.4f}"
                    )

                # update progress bar + table in one call
                progress.advance_and_update(attack_task, values, advance=1)
                progress.refresh()
                save_adv_images = cfg.get("save_adv_images", False)
                save_adv_predictions = cfg.get("save_adv_predictions", False)
                if save_adv_images or save_adv_predictions:
                    for j in range(images.shape[0]):
                        image_path = image_list[i * batch_size + j]
                        image_name = image_path.split("/")[-1].split(".")[0]
                        if save_adv_images:
                            check_makedirs(cfg.image_adv_dir)
                            scale = 255 if cfg.data.normalize else 1.0
                            orig_image = cv2.cvtColor(
                                np.uint8((images[j] * scale).cpu().numpy()).transpose(
                                    1, 2, 0
                                ),
                                cv2.COLOR_RGB2BGR,
                            )
                            adv_image = cv2.cvtColor(
                                np.uint8(
                                    (images_adv[j] * scale).cpu().numpy()
                                ).transpose(1, 2, 0),
                                cv2.COLOR_RGB2BGR,
                            )
                            image_path = os.path.join(
                                cfg.image_adv_dir, image_name + ".png"
                            )
                            adv_image_path = os.path.join(
                                cfg.image_adv_dir, image_name + "_adv.png"
                            )
                            cv2.imwrite(image_path, orig_image)
                            cv2.imwrite(adv_image_path, adv_image)
                        if save_adv_predictions:
                            check_makedirs(cfg.gray_dir)
                            check_makedirs(cfg.color_dir)
                            gray = np.uint8(pred_orig[j].cpu().numpy())
                            color = colorize(gray, colormap)
                            gray_adv = np.uint8(pred_adv[j].cpu().numpy())
                            color_adv = colorize(gray_adv, colormap)
                            gray_path = os.path.join(cfg.gray_dir, image_name + ".png")
                            color_path = os.path.join(
                                cfg.color_dir, image_name + ".png"
                            )
                            # original predictions
                            cv2.imwrite(gray_path, gray)
                            color.save(color_path)
                            # adversarial predictions
                            gray_adv_path = os.path.join(
                                cfg.gray_dir, image_name + "_adv.png"
                            )
                            color_adv_path = os.path.join(
                                cfg.color_dir, image_name + "_adv.png"
                            )
                            cv2.imwrite(gray_adv_path, gray_adv)
                            color_adv.save(color_adv_path)
    except KeyboardInterrupt:
        log.info(f"Attack interrupted by user at batch {i}.")

    # compute final metrics
    acc_global_orig, accs_orig, ious_orig = confusion_matrix_orig.compute()
    acc_global_adv, accs_adv, ious_adv = confusion_matrix_adv.compute()

    final_metrics = OrderedDict()
    final_metrics["acc_orig"] = acc_global_orig.item()
    final_metrics["macc_orig"] = accs_orig.nanmean().item()
    final_metrics["acc_adv"] = acc_global_adv.item()
    final_metrics["miou_orig"] = ious_orig.nanmean().item()
    final_metrics["macc_adv"] = accs_adv.nanmean().item()
    final_metrics["miou_adv"] = ious_adv.nanmean().item()
    for iteration in cfg.return_at_iterations:
        acc_global_adv_i, accs_adv_i, ious_adv_i = confusion_matrix_at_iter[
            iteration
        ].compute()
        final_metrics[f"acc_adv_{iteration}"] = acc_global_adv_i.item()
        final_metrics[f"macc_adv_{iteration}"] = accs_adv_i.nanmean().item()
        final_metrics[f"miou_adv_{iteration}"] = ious_adv_i.nanmean().item()

    if class_names is not None:
        log.info("Per-class accuracies (orig | adv):")
        width = max(len(name) for name in class_names)
        for idx, name in enumerate(class_names[: len(accs_orig)]):
            acc_o = accs_orig[idx].item()
            acc_a = accs_adv[idx].item()
            log.info(
                f"[{idx + dataset.get_label_offset():03d}] {name.ljust(width)} | {acc_o:0.4f} | {acc_a:0.4f}"
            )
    else:
        log.info(
            "Per-class accuracies not printed because class names are unavailable."
        )

    for logger in loggers:
        logger.log_metrics(final_metrics)
        logger.save()

    return metrics_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="test.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    if torch.cuda.is_available():
        import torch.backends.cudnn as cudnn

        cudnn.benchmark = True
        cudnn.deterministic = False
        torch.set_float32_matmul_precision("high")
    else:
        log.warning("CUDA is not available. Exiting...")
        exit(0)
    assert cfg.job_id is not None and cfg.job_id >= 0, "job_id must be >= 0"
    if cfg.seed_random:
        np.random.seed(cfg.seed)
        for _ in range(cfg.job_id + 1):
            seed = np.random.randint(0, 2**32)
        cfg.seed = seed
    if cfg.return_at_iterations is None or cfg.return_at_iterations == "[]":
        cfg.return_at_iterations = []
    check_existing_run_dir(cfg)
    validate_enum_value(cfg.data_split, DataSplit)
    setup_extras(cfg)

    metric_dict, _ = run_attack(cfg)


if __name__ == "__main__":
    register_configs()
    setup_resolvers()
    main()
