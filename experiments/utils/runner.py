import os
import re
import shutil
import sys
import warnings
from functools import partial
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

from omegaconf import DictConfig

from .pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

# Constants
CONFIG_EXTRAS = "extras"
CONFIG_IGNORE_WARNINGS = "ignore_warnings"
CONFIG_ENFORCE_TAGS = "enforce_tags"
CONFIG_PRINT_CONFIG = "print_config"


def _handle_warnings(cfg: DictConfig) -> None:
    """Handle Python warnings based on configuration."""
    if cfg[CONFIG_EXTRAS].get(CONFIG_IGNORE_WARNINGS):
        log.info(
            f"Disabling python warnings! <cfg.{CONFIG_EXTRAS}.{CONFIG_IGNORE_WARNINGS}=True>"
        )
        warnings.filterwarnings("ignore")


def _handle_tags(cfg: DictConfig) -> None:
    """Handle tag enforcement based on configuration."""
    if cfg[CONFIG_EXTRAS].get(CONFIG_ENFORCE_TAGS):
        from .rich import enforce_tags

        log.info(f"Enforcing tags! <cfg.{CONFIG_EXTRAS}.{CONFIG_ENFORCE_TAGS}=True>")
        enforce_tags(cfg, save_to_file=True)


def _print_config(cfg: DictConfig) -> None:
    """Print configuration tree based on configuration."""
    if cfg[CONFIG_EXTRAS].get(CONFIG_PRINT_CONFIG):
        from .rich import print_config_tree

        log.info(
            f"Printing config tree with Rich! <cfg.{CONFIG_EXTRAS}.{CONFIG_PRINT_CONFIG}=True>"
        )
        print_config_tree(cfg, resolve=True, save_to_file=True)


def check_makedir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


def check_makedirs(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def check_existing_run_dir(cfg):
    """Checks if a run directory with the same run_name already exists in the
    output parent directory.

    Warns when a prior run with the same base name exists. Hydra output
    directories are timestamped, so a previous run should not block a rerun.

    When ``cfg.overwrite`` is True the check is skipped entirely.
    """
    overwrite = cfg.get("overwrite", False)
    parent = Path(cfg.paths.output_dir).parent
    base_run_name = re.split(r"-\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$", cfg.run_name)[0]
    for dir in parent.iterdir():
        run_name = dir.name
        if (
            dir.is_dir()
            and (dir / "config_tree.log").is_file()
            and run_name.startswith(base_run_name)
        ):
            if overwrite:
                log.warning(
                    f"Overwrite enabled — removing existing run directory: '{dir}'"
                )
                shutil.rmtree(dir)
            else:
                log.warning(
                    f"Output directory for a run '{cfg.run_name}' already exists: '{dir}'. "
                )
                sys.exit(1)


def setup_extras(cfg: DictConfig) -> None:
    """Applies optional utilities before the task is started."""

    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    _handle_warnings(cfg)
    _handle_tags(cfg)
    _print_config(cfg)


def setup_paths(cfg: DictConfig) -> None:
    """Create the directories specified in the configuration."""

    if not cfg.get("paths"):
        log.warning("Paths config not found! <cfg.paths=null>")
        return

    checkpoints_dir = cfg.paths.get("checkpoints_dir")
    if checkpoints_dir:
        log.info(f"Creating checkpoints directory: {checkpoints_dir}")
        try:
            os.makedirs(checkpoints_dir, exist_ok=True)
        except OSError as e:
            log.error(f"Failed to create checkpoints directory {checkpoints_dir}: {e}")


def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that controls the failure behavior when executing the
    task function."""

    def wrap(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        try:
            results = task_func(cfg=cfg)
        except Exception as ex:
            log.exception("")
            raise ex

        finally:
            log.info(f"Output dir: {cfg.paths.output_dir}")
            if find_spec("wandb"):  # check if wandb is installed
                import wandb

                if wandb.run:
                    log.info("Closing wandb!")
                    wandb.finish()

        return results

    return wrap


def resolve_fn_name(fn: Callable) -> str:
    """Resolve a human-readable name from a callable.

    Handles ``nn.Module`` subclasses, ``functools.partial`` wrappers, and
    plain functions/lambdas.
    """
    from torch import nn

    if isinstance(fn, nn.Module):
        return type(fn).__name__
    if isinstance(fn, partial):
        return resolve_fn_name(fn.func)
    return getattr(fn, "__name__", type(fn).__name__)
