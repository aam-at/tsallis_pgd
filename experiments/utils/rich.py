from pathlib import Path
from typing import Dict, List, Sequence

import rich
import rich.console
import rich.syntax
import rich.tree
from hydra.core.hydra_config import HydraConfig
from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import DictConfig, OmegaConf, open_dict
from rich import box
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.prompt import Prompt
from rich.table import Table

from .pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


@rank_zero_only
def print_config_tree(
    cfg: DictConfig,
    print_order: Sequence[str] = (
        "data",
        "model",
        "callbacks",
        "logger",
        "trainer",
        "paths",
        "extras",
    ),
    resolve: bool = False,
    save_to_file: bool = False,
) -> None:
    """Print the config tree, using Rich when available."""
    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)
    queue = [field for field in print_order if field in cfg] + [
        field for field in cfg if field not in print_order
    ]

    for field in queue:
        branch = tree.add(field, style=style, guide_style=style)
        config_group = cfg[field]
        if isinstance(config_group, DictConfig):
            branch_content = OmegaConf.to_yaml(config_group, resolve=resolve)
        else:
            branch_content = str(config_group)
        branch.add(rich.syntax.Syntax(branch_content, "yaml"))
    rich.print(tree)
    if save_to_file:
        with open(Path(cfg.paths.output_dir, "config_tree.log"), "w") as file:
            rich.print(tree, file=file)


@rank_zero_only
def enforce_tags(cfg: DictConfig, save_to_file: bool = False) -> None:
    """Prompt for tags when possible, otherwise fall back to a default tag."""
    if not cfg.get("tags"):
        if "id" in HydraConfig().cfg.hydra.job:
            raise ValueError("Specify tags before launching a multirun!")

        log.warning("No tags provided in config.")
        tags = Prompt.ask("Enter a list of comma separated tags", default="dev")
        tags = [t.strip() for t in tags.split(",") if t != ""]

        with open_dict(cfg):
            cfg.tags = tags

        log.info(f"Tags set to: {cfg.tags}")

    if save_to_file:
        content = "\n".join(cfg.tags)
        Path(cfg.paths.output_dir, "tags.log").write_text(content)


class AttackProgress(Progress):
    """Enhanced Progress bar with a customizable metrics table."""

    def __init__(
        self,
        metrics: List[str] = None,
        categories: List[str] = None,
        table_box: box = box.SIMPLE,
        metric_style: str = "cyan",
        category_justification: str = "right",
        initial_values: Dict[str, Dict[str, str]] = None,
        precision: int = 4,
        *args,
        **kwargs,
    ) -> None:
        self.metrics = metrics or ["Accuracy", "mIoU"]
        self.categories = categories or ["Original", "Adversarial"]
        self.table_box = table_box
        self.metric_style = metric_style
        self.category_justification = category_justification
        self.precision = precision

        self.values = {}
        default_value = "0.0000"

        for metric in self.metrics:
            self.values[metric] = {}
            for category in self.categories:
                if (
                    initial_values
                    and metric in initial_values
                    and category in initial_values[metric]
                ):
                    self.values[metric][category] = initial_values[metric][category]
                else:
                    self.values[metric][category] = default_value

        self._dirty = False
        self._create_table()
        # Add batch count and ETA columns when no custom columns are provided
        if not args:
            columns = (
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TaskProgressColumn(),
                TextColumn("elapsed"),
                TimeElapsedColumn(),
                TextColumn("eta"),
                TimeRemainingColumn(),
            )
            kwargs.setdefault("speed_estimate_period", 3600)
            super().__init__(*columns, **kwargs)
        else:
            super().__init__(*args, **kwargs)

    def _create_table(self) -> None:
        table = Table(box=self.table_box)
        table.add_column("Metric", style=self.metric_style)
        for category in self.categories:
            table.add_column(category, justify=self.category_justification)
        for metric in self.metrics:
            row_values = [metric]
            for category in self.categories:
                row_values.append(self.values[metric][category])
            table.add_row(*row_values)
        self.table = table
        self._dirty = False

    def _format_value(self, value: object) -> str:
        if isinstance(value, (float, int)):
            return f"{float(value):.{self.precision}f}"
        return str(value)

    def _set_value(
        self,
        metric: str,
        category: str,
        value: object,
        add_missing: bool = False,
    ) -> None:
        if metric not in self.metrics:
            if add_missing:
                self.metrics.append(metric)
                self.values[metric] = {
                    existing_category: "0.0000" for existing_category in self.categories
                }
            else:
                raise ValueError(f"Invalid metric '{metric}'")
        if category not in self.categories:
            if add_missing:
                self.categories.append(category)
                for existing_metric in self.metrics:
                    self.values.setdefault(existing_metric, {})
                    self.values[existing_metric].setdefault(category, "0.0000")
            else:
                raise ValueError(f"Invalid category '{category}'")
        self.values[metric][category] = self._format_value(value)
        self._dirty = True

    def update_all(
        self, values: Dict[str, Dict[str, object]], add_missing: bool = False
    ) -> None:
        for metric, category_values in values.items():
            for category, value in category_values.items():
                try:
                    self._set_value(metric, category, value, add_missing=add_missing)
                except ValueError:
                    continue

    def advance_and_update(
        self,
        task_id: int,
        values: Dict[str, Dict[str, object]],
        advance: int = 1,
    ) -> None:
        self.advance(task_id, advance=advance)
        self.update_all(values, add_missing=True)
        if self._dirty:
            self._create_table()

    def get_renderable(self):
        if self._dirty:
            self._create_table()
        progress_renderable = super().get_renderable()
        return rich.console.Group(progress_renderable, self.table)
