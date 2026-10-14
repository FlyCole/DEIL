#!/usr/bin/env python3
"""Create metric and success diagnostics from the IP rollout workbook.

The script reads the XLSX produced by ``process_ip_rollout_artifacts.py``
without requiring pandas or openpyxl. It produces:

1. configuration-level metric value versus success rate;
2. per-task empirical success-rate curves binned by rollout metric value;
3. per-task metric distributions split by successful/failed rollouts.

These views are generated for every metric recorded in the workbook by default.
The script also writes a rollout-weighted per-estimator CSV summary table.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter
import numpy as np
from scipy.stats import spearmanr


MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PACKAGE_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WORKBOOK = REPOSITORY_ROOT / "results" / "ip_eval" / "ip_rollout_results.xlsx"
DEFAULT_EXPECTED_ROLLOUTS = 200
CONFIG_FIELDS = (
    "task_name",
    "depth_estimator",
    "camera_names",
    "resolution",
    "mask_input",
    "eval_name",
)


@dataclass(frozen=True)
class MetricSpec:
    name: str
    display_name: str
    units: str
    lower_is_better: bool

    @property
    def column(self) -> str:
        return f"{self.name}_mean"

    @property
    def file_name(self) -> str:
        return "".join(
            character if character.isalnum() or character in "-_" else "_"
            for character in self.name
        )

    @property
    def direction_text(self) -> str:
        return "lower is better" if self.lower_is_better else "higher is better"

    @property
    def axis_label(self) -> str:
        units = f" ({self.units})" if self.units else ""
        return f"Rollout mean {self.display_name}{units}"


@dataclass(frozen=True)
class ConfigSummary:
    key: Tuple[str, str, str, int, str, str]
    task_name: str
    estimator: str
    camera_names: str
    resolution: int
    mask_input: str
    eval_name: str
    rollout_count: int
    expected_rollouts: int
    metric_mean: float
    metric_std: float
    success_rate: float
    is_complete: bool

    @property
    def series_label(self) -> str:
        cameras = self.camera_names.replace("|", "+")
        return f"{display_name(self.estimator)} ({cameras}, {self.resolution})"


def display_name(value: str) -> str:
    text = str(value)
    mapping = {
        "gt": "GT depth",
        "map_anything": "MapAnything",
        "mast3r": "MASt3R",
        "moge": "MoGe",
        "pi3": r"$\pi^3$",
        "vggt": "VGGT",
    }
    return mapping.get(text.casefold(), text.replace("_", " ").title())


def task_display_name(value: str) -> str:
    return str(value).replace("_", " ").title()


def _column_index(reference: str) -> int:
    letters = "".join(character for character in reference if character.isalpha())
    index = 0
    for character in letters:
        index = index * 26 + ord(character.upper()) - 64
    return index - 1


def _shared_strings(archive: zipfile.ZipFile) -> list[str]:
    try:
        root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    except KeyError:
        return []
    strings = []
    for item in root.findall(f"{{{MAIN_NS}}}si"):
        strings.append("".join(node.text or "" for node in item.iter(f"{{{MAIN_NS}}}t")))
    return strings


def _worksheet_path(archive: zipfile.ZipFile, sheet_name: str) -> str:
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    targets = {
        relationship.attrib["Id"]: relationship.attrib["Target"]
        for relationship in relationships.findall(
            f"{{{PACKAGE_REL_NS}}}Relationship"
        )
    }
    for sheet in workbook.findall(f".//{{{MAIN_NS}}}sheet"):
        if sheet.attrib.get("name") != sheet_name:
            continue
        target = targets[sheet.attrib[f"{{{REL_NS}}}id"]].lstrip("/")
        if target.startswith("xl/"):
            return target
        return f"xl/{target}"
    raise KeyError(f"Workbook has no worksheet named {sheet_name!r}")


def read_xlsx_sheet(path: Path, sheet_name: str) -> list[Dict[str, Any]]:
    """Read one worksheet produced by the IP processor."""
    with zipfile.ZipFile(path) as archive:
        shared_strings = _shared_strings(archive)
        worksheet = ET.fromstring(
            archive.read(_worksheet_path(archive, sheet_name))
        )

    raw_rows = []
    for row in worksheet.findall(f".//{{{MAIN_NS}}}row"):
        values: Dict[int, Any] = {}
        for cell in row.findall(f"{{{MAIN_NS}}}c"):
            column = _column_index(cell.attrib["r"])
            cell_type = cell.attrib.get("t")
            if cell_type == "inlineStr":
                value = "".join(
                    node.text or ""
                    for node in cell.findall(f".//{{{MAIN_NS}}}t")
                )
            else:
                value_node = cell.find(f"{{{MAIN_NS}}}v")
                if value_node is None or value_node.text is None:
                    continue
                raw_value = value_node.text
                if cell_type == "s":
                    value = shared_strings[int(raw_value)]
                elif cell_type == "b":
                    value = raw_value == "1"
                elif cell_type == "str":
                    value = raw_value
                else:
                    value = float(raw_value)
            values[column] = value
        raw_rows.append(values)

    if not raw_rows:
        return []
    headers = {column: str(value) for column, value in raw_rows[0].items()}
    return [
        {
            headers[column]: value
            for column, value in row.items()
            if column in headers
        }
        for row in raw_rows[1:]
    ]


def _text(row: Dict[str, Any], key: str, default="") -> str:
    value = row.get(key, default)
    if value is None:
        return str(default)
    return str(value)


def _number(row: Dict[str, Any], key: str) -> Optional[float]:
    value = row.get(key)
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _boolean(row: Dict[str, Any], key: str) -> Optional[bool]:
    value = row.get(key)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().casefold()
    if text in {"true", "1", "yes", "success"}:
        return True
    if text in {"false", "0", "no", "failure"}:
        return False
    return None


def _config_key(row: Dict[str, Any]) -> Tuple[str, str, str, int, str, str]:
    resolution = _number(row, "resolution")
    return (
        _text(row, "task_name"),
        _text(row, "depth_estimator"),
        _text(row, "camera_names"),
        int(resolution) if resolution is not None else 0,
        _text(row, "mask_input", "none"),
        _text(row, "eval_name", "eval_0000"),
    )


def _lower_is_better(metric_name: str) -> bool:
    return not metric_name.startswith(("precision_", "recall_", "fscore_"))


def load_metric_specs(
    workbook: Path,
    requested: Optional[Sequence[str]] = None,
) -> list[MetricSpec]:
    """Load plot labels and units from the processor's metric registry sheet."""
    definition_rows = read_xlsx_sheet(workbook, "Metric Definitions")
    available = {}
    for row in definition_rows:
        name = _text(row, "metric_name").strip()
        if not name:
            continue
        available[name] = MetricSpec(
            name=name,
            display_name=_text(row, "display_name", display_name(name)),
            units=_text(row, "units").strip(),
            lower_is_better=_lower_is_better(name),
        )
    if not available:
        raise RuntimeError("The workbook contains no metric definitions")

    names = list(requested) if requested else list(available)
    if len(names) != len(set(names)):
        raise ValueError("--metrics contains duplicate metric names")
    missing = [name for name in names if name not in available]
    if missing:
        raise ValueError(
            f"Unknown metric(s) {missing}; available metrics: "
            f"{sorted(available)}"
        )
    return [available[name] for name in names]


def load_rollout_rows(
    workbook: Path,
    tasks: Optional[set[str]] = None,
    estimators: Optional[set[str]] = None,
) -> Tuple[list[Dict[str, Any]], Dict[Tuple[str, ...], int]]:
    rollout_rows = read_xlsx_sheet(workbook, "Rollouts")
    evaluation_rows = read_xlsx_sheet(workbook, "Evaluations")
    expected_by_config = {}
    for row in evaluation_rows:
        key = _config_key(row)
        expected = _number(row, "expected_rollouts")
        if expected is not None and expected > 0:
            expected_by_config[key] = int(expected)

    filtered = []
    for row in rollout_rows:
        if _text(row, "eval_name") != "eval_0000":
            continue
        task = _text(row, "task_name")
        estimator = _text(row, "depth_estimator")
        if tasks is not None and task not in tasks:
            continue
        if estimators is not None and estimator not in estimators:
            continue
        success = _boolean(row, "success")
        if success is None:
            continue
        normalized = dict(row)
        normalized["success"] = success
        filtered.append(normalized)
    return filtered, expected_by_config


def summarize_configurations(
    rollout_rows: Sequence[Dict[str, Any]],
    expected_by_config: Dict[Tuple[str, ...], int],
    metric: MetricSpec,
) -> Tuple[list[ConfigSummary], Dict[Tuple[str, ...], list[Dict[str, Any]]]]:
    all_grouped: Dict[Tuple[str, ...], list[Dict[str, Any]]] = {}
    for row in rollout_rows:
        all_grouped.setdefault(_config_key(row), []).append(row)

    summaries = []
    metric_grouped = {}
    for key, all_rows in sorted(all_grouped.items()):
        rows = []
        for row in all_rows:
            value = _number(row, metric.column)
            if value is None:
                continue
            normalized = dict(row)
            normalized[metric.column] = value
            rows.append(normalized)
        if not rows:
            continue
        metric_grouped[key] = rows
        values = np.asarray([row[metric.column] for row in rows], dtype=float)
        success = np.asarray([row["success"] for row in rows], dtype=float)
        expected = expected_by_config.get(key, DEFAULT_EXPECTED_ROLLOUTS)
        rollout_ids = {
            int(value)
            for row in all_rows
            if (value := _number(row, "rollout_id")) is not None
        }
        is_complete = (
            len(all_rows) == expected
            and len(rows) == len(all_rows)
            and rollout_ids == set(range(expected))
        )
        task, estimator, cameras, resolution, mask_input, eval_name = key
        summaries.append(ConfigSummary(
            key=key,
            task_name=task,
            estimator=estimator,
            camera_names=cameras,
            resolution=resolution,
            mask_input=mask_input,
            eval_name=eval_name,
            rollout_count=len(rows),
            expected_rollouts=expected,
            metric_mean=float(values.mean()),
            metric_std=float(values.std()),
            success_rate=float(success.mean()),
            is_complete=is_complete,
        ))
    return summaries, metric_grouped


def _color_map(labels: Iterable[str]) -> Dict[str, Any]:
    labels = sorted(set(labels))
    color_map = plt.get_cmap("tab10")
    return {label: color_map(index % 10) for index, label in enumerate(labels)}


def _save_figure(
    figure: plt.Figure,
    output_dir: Path,
    stem: str,
    formats: Sequence[str],
    dpi: int,
) -> list[Path]:
    paths = []
    for extension in formats:
        path = output_dir / f"{stem}.{extension}"
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
        paths.append(path)
    plt.close(figure)
    return paths


def plot_global_metric_vs_success(
    summaries: Sequence[ConfigSummary],
    metric: MetricSpec,
    output_dir: Path,
    formats: Sequence[str],
    dpi: int,
    annotate: bool,
    log_x: bool,
) -> Tuple[list[Path], Dict[str, Optional[float]]]:
    if not summaries:
        raise ValueError("No configuration summaries are available for plotting")
    colors = _color_map(summary.estimator for summary in summaries)
    tasks = sorted({summary.task_name for summary in summaries})
    marker_names = ("o", "s", "^", "D", "v", "P", "X", "<", ">", "h", "*")
    markers = {
        task: marker_names[index % len(marker_names)]
        for index, task in enumerate(tasks)
    }

    figure, axis = plt.subplots(figsize=(9.5, 6.5), constrained_layout=True)
    for summary in summaries:
        n = max(summary.rollout_count, 1)
        x_error = 1.96 * summary.metric_std / math.sqrt(n)
        p = summary.success_rate
        y_error = 1.96 * math.sqrt(max(p * (1.0 - p), 0.0) / n)
        axis.errorbar(
            summary.metric_mean,
            summary.success_rate,
            xerr=x_error,
            yerr=y_error,
            fmt=markers[summary.task_name],
            markersize=8,
            color=colors[summary.estimator],
            ecolor=colors[summary.estimator],
            elinewidth=1.0,
            capsize=2.5,
            alpha=0.9 if summary.is_complete else 0.45,
            markerfacecolor=(
                colors[summary.estimator] if summary.is_complete else "none"
            ),
        )
        if annotate:
            axis.annotate(
                task_display_name(summary.task_name),
                (summary.metric_mean, summary.success_rate),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )

    x_values = np.asarray([summary.metric_mean for summary in summaries])
    y_values = np.asarray([summary.success_rate for summary in summaries])
    pearson = None
    spearman = None
    if (
        len(summaries) >= 2
        and float(np.ptp(x_values)) > 0
        and float(np.ptp(y_values)) > 0
    ):
        pearson = float(np.corrcoef(x_values, y_values)[0, 1])
        spearman = float(spearmanr(x_values, y_values).statistic)
        slope, intercept = np.polyfit(x_values, y_values, 1)
        line_x = np.linspace(float(x_values.min()), float(x_values.max()), 200)
        axis.plot(
            line_x,
            np.clip(slope * line_x + intercept, 0.0, 1.0),
            linestyle="--",
            color="#333333",
            linewidth=1.4,
        )
        axis.text(
            0.015,
            0.985,
            f"Pearson r={pearson:.2f}\nSpearman ρ={spearman:.2f}",
            transform=axis.transAxes,
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.85},
        )

    estimator_handles = [
        Line2D(
            [0], [0], marker="o", linestyle="none", markersize=8,
            markerfacecolor=color, markeredgecolor=color,
            label=display_name(estimator),
        )
        for estimator, color in colors.items()
    ]
    if pearson is not None:
        estimator_handles.append(Line2D(
            [0], [0], linestyle="--", color="#333333",
            label="Linear trend",
        ))
    if estimator_handles:
        estimator_legend = axis.legend(
            handles=estimator_handles,
            title="Depth estimator",
            loc="best",
        )
        axis.add_artist(estimator_legend)
    if len(tasks) <= 12 and not annotate:
        task_handles = [
            Line2D(
                [0], [0], marker=markers[task], linestyle="none",
                color="#444444", markersize=7,
                label=task_display_name(task),
            )
            for task in tasks
        ]
        axis.legend(handles=task_handles, title="Task", loc="upper right")

    if log_x:
        if np.any(x_values <= 0):
            raise ValueError(
                f"--log_x requires positive {metric.name} values"
            )
        axis.set_xscale("log")
    axis.set_xlabel(f"{metric.axis_label} ({metric.direction_text})")
    if metric.units == "fraction":
        axis.xaxis.set_major_formatter(PercentFormatter(1.0))
    axis.set_ylabel("Success rate")
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.set_ylim(-0.04, 1.04)
    axis.grid(True, alpha=0.25)
    axis.set_title(f"{metric.display_name} vs. Task Success Rate")
    paths = _save_figure(
        figure,
        output_dir,
        f"{metric.file_name}_vs_success_rate",
        formats,
        dpi,
    )
    return paths, {"pearson_r": pearson, "spearman_r": spearman}


def _quantile_success_curve(
    rows: Sequence[Dict[str, Any]],
    metric: MetricSpec,
    bin_count: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray([row[metric.column] for row in rows], dtype=float)
    success = np.asarray([row["success"] for row in rows], dtype=float)
    edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, bin_count + 1)))
    if len(edges) < 2:
        return tuple(np.asarray([], dtype=float) for _ in range(5))
    bin_ids = np.searchsorted(edges[1:-1], values, side="right")
    centers = []
    rates = []
    lower_errors = []
    upper_errors = []
    counts = []
    z_value = 1.96
    for bin_id in range(len(edges) - 1):
        selected = bin_ids == bin_id
        count = int(selected.sum())
        if count == 0:
            continue
        center = float(np.median(values[selected]))
        rate = float(success[selected].mean())
        denominator = 1.0 + z_value ** 2 / count
        interval_center = (
            rate + z_value ** 2 / (2.0 * count)
        ) / denominator
        half_width = (
            z_value
            * math.sqrt(
                rate * (1.0 - rate) / count
                + z_value ** 2 / (4.0 * count ** 2)
            )
            / denominator
        )
        lower = max(0.0, interval_center - half_width)
        upper = min(1.0, interval_center + half_width)
        centers.append(center)
        rates.append(rate)
        lower_errors.append(rate - lower)
        upper_errors.append(upper - rate)
        counts.append(count)
    return (
        np.asarray(centers),
        np.asarray(rates),
        np.asarray(lower_errors),
        np.asarray(upper_errors),
        np.asarray(counts),
    )


def plot_task_success_curves(
    summaries: Sequence[ConfigSummary],
    grouped_rows: Dict[Tuple[str, ...], list[Dict[str, Any]]],
    metric: MetricSpec,
    output_dir: Path,
    formats: Sequence[str],
    dpi: int,
    bin_count: int,
    log_x: bool,
) -> list[Path]:
    tasks = sorted({summary.task_name for summary in summaries})
    if not tasks:
        raise ValueError("No tasks are available for task-specific plotting")
    column_count = min(3, len(tasks))
    row_count = math.ceil(len(tasks) / column_count)
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(5.2 * column_count, 4.0 * row_count),
        squeeze=False,
        constrained_layout=True,
    )
    colors = plt.get_cmap("tab10")
    random = np.random.default_rng(0)
    for task_index, task in enumerate(tasks):
        axis = axes.flat[task_index]
        task_summaries = [
            summary for summary in summaries if summary.task_name == task
        ]
        for series_index, summary in enumerate(task_summaries):
            rows = grouped_rows[summary.key]
            x_values = np.asarray([row[metric.column] for row in rows])
            success = np.asarray([row["success"] for row in rows], dtype=float)
            jitter = random.normal(0.0, 0.015, size=len(rows))
            color = colors(series_index % 10)
            axis.scatter(
                x_values,
                np.clip(success + jitter, -0.03, 1.03),
                s=8,
                alpha=0.08,
                color=color,
                rasterized=True,
            )
            centers, rates, lower, upper, counts = _quantile_success_curve(
                rows,
                metric,
                bin_count,
            )
            if len(centers) == 0:
                continue
            label = summary.series_label
            if len(task_summaries) == 1:
                label = f"Binned SR (n={summary.rollout_count})"
            axis.errorbar(
                centers,
                rates,
                yerr=np.vstack((lower, upper)),
                marker="o",
                markersize=5,
                linewidth=1.8,
                capsize=3,
                color=color,
                label=label,
            )
            for x_value, y_value, count in zip(centers, rates, counts):
                axis.annotate(
                    f"n={int(count)}",
                    (x_value, y_value),
                    xytext=(0, 7),
                    textcoords="offset points",
                    ha="center",
                    fontsize=6,
                    color=color,
                )

        if log_x:
            has_nonpositive = any(
                row[metric.column] <= 0
                for summary in task_summaries
                for row in grouped_rows[summary.key]
            )
            if has_nonpositive:
                raise ValueError(
                    f"--log_x requires positive {metric.name} values"
                )
            axis.set_xscale("log")
        if metric.units == "fraction":
            axis.xaxis.set_major_formatter(PercentFormatter(1.0))
        axis.set_ylim(-0.08, 1.08)
        axis.yaxis.set_major_formatter(PercentFormatter(1.0))
        axis.set_title(task_display_name(task))
        axis.set_xlabel(metric.axis_label)
        axis.set_ylabel("Empirical success rate")
        axis.grid(True, alpha=0.22)
        legend_handles, legend_labels = axis.get_legend_handles_labels()
        if legend_handles:
            axis.legend(
                legend_handles,
                legend_labels,
                fontsize=8,
                loc="best",
            )

    for empty_index in range(len(tasks), row_count * column_count):
        figure.delaxes(axes.flat[empty_index])
    figure.suptitle(
        f"Task-Specific Success Rate vs. {metric.display_name}\n"
        "Quantile bins with 95% Wilson confidence intervals",
        fontsize=14,
    )
    return _save_figure(
        figure,
        output_dir,
        f"task_{metric.file_name}_vs_success_rate",
        formats,
        dpi,
    )


def plot_success_failure_distributions(
    summaries: Sequence[ConfigSummary],
    grouped_rows: Dict[Tuple[str, ...], list[Dict[str, Any]]],
    metric: MetricSpec,
    output_dir: Path,
    formats: Sequence[str],
    dpi: int,
    log_x: bool,
) -> list[Path]:
    tasks = sorted({summary.task_name for summary in summaries})
    success_data = []
    success_positions = []
    failure_data = []
    failure_positions = []
    task_counts = {}
    for task_index, task in enumerate(tasks):
        rows = [
            row
            for summary in summaries
            if summary.task_name == task
            for row in grouped_rows[summary.key]
        ]
        successful = [row[metric.column] for row in rows if row["success"]]
        failed = [row[metric.column] for row in rows if not row["success"]]
        task_counts[task] = (len(successful), len(failed))
        if successful:
            success_data.append(successful)
            success_positions.append(task_index + 0.18)
        if failed:
            failure_data.append(failed)
            failure_positions.append(task_index - 0.18)

    figure, axis = plt.subplots(
        figsize=(10.5, max(5.0, 0.62 * len(tasks) + 2.0)),
        constrained_layout=True,
    )
    if failure_data:
        failure_boxes = axis.boxplot(
            failure_data,
            positions=failure_positions,
            widths=0.28,
            vert=False,
            patch_artist=True,
            showfliers=False,
            manage_ticks=False,
        )
        for box in failure_boxes["boxes"]:
            box.set_facecolor("#D95F5F")
            box.set_alpha(0.7)
    if success_data:
        success_boxes = axis.boxplot(
            success_data,
            positions=success_positions,
            widths=0.28,
            vert=False,
            patch_artist=True,
            showfliers=False,
            manage_ticks=False,
        )
        for box in success_boxes["boxes"]:
            box.set_facecolor("#4C9F70")
            box.set_alpha(0.75)

    labels = []
    for task in tasks:
        success_count, failure_count = task_counts[task]
        labels.append(
            f"{task_display_name(task)}  (S={success_count}, F={failure_count})"
        )
    axis.set_yticks(range(len(tasks)), labels)
    if log_x:
        all_values = [
            value
            for values in success_data + failure_data
            for value in values
        ]
        if any(value <= 0 for value in all_values):
            raise ValueError(
                f"--log_x requires positive {metric.name} values"
            )
        axis.set_xscale("log")
    if metric.units == "fraction":
        axis.xaxis.set_major_formatter(PercentFormatter(1.0))
    axis.set_xlabel(f"{metric.axis_label} ({metric.direction_text})")
    association_text = (
        "Lower successful-rollout boxes indicate association with success"
        if metric.lower_is_better
        else "Higher successful-rollout boxes indicate association with success"
    )
    axis.set_title(
        f"{metric.display_name} Distribution Conditioned on Task Outcome\n"
        f"{association_text}"
    )
    axis.grid(True, axis="x", alpha=0.25)
    axis.legend(
        handles=[
            Patch(facecolor="#4C9F70", alpha=0.75, label="Successful rollout"),
            Patch(facecolor="#D95F5F", alpha=0.7, label="Failed rollout"),
        ],
        loc="best",
    )
    return _save_figure(
        figure,
        output_dir,
        f"success_failure_{metric.file_name}_by_task",
        formats,
        dpi,
    )


def write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.stem}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as output_file:
            temp_path = Path(output_file.name)
            json.dump(payload, output_file, indent=2, sort_keys=True)
            output_file.write("\n")
        temp_path.replace(path)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def build_estimator_summary_rows(
    rollout_rows: Sequence[Dict[str, Any]],
    metric_specs: Sequence[MetricSpec],
) -> list[Dict[str, Any]]:
    """Aggregate SR and rollout-level metric means for each estimator."""
    grouped = {}
    for row in rollout_rows:
        grouped.setdefault(_text(row, "depth_estimator"), []).append(row)

    table_rows = []
    for estimator, rows in sorted(grouped.items()):
        successes = np.asarray([row["success"] for row in rows], dtype=float)
        table_row = {
            "depth_estimator": estimator,
            "display_name": display_name(estimator),
            "configuration_count": len({_config_key(row) for row in rows}),
            "rollout_count": len(rows),
            "success_count": int(successes.sum()),
            "success_rate": float(successes.mean()),
            "success_rate_percent": float(successes.mean() * 100.0),
        }
        for metric in metric_specs:
            values = [
                value
                for row in rows
                if (value := _number(row, metric.column)) is not None
            ]
            table_row[f"{metric.name}_rollout_count"] = len(values)
            table_row[metric.column] = (
                float(np.mean(values)) if values else None
            )
        table_rows.append(table_row)
    return table_rows


def write_estimator_summary_csv(
    path: Path,
    rows: Sequence[Dict[str, Any]],
    metric_specs: Sequence[MetricSpec],
) -> None:
    headers = [
        "depth_estimator",
        "display_name",
        "configuration_count",
        "rollout_count",
        "success_count",
        "success_rate",
        "success_rate_percent",
    ]
    for metric in metric_specs:
        headers.extend((f"{metric.name}_rollout_count", metric.column))

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            prefix=f".{path.stem}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as output_file:
            temp_path = Path(output_file.name)
            writer = csv.DictWriter(output_file, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)
        temp_path.replace(path)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Plot point-cloud metrics and success diagnostics from IP summaries."
    )
    parser.add_argument("--workbook", type=Path, default=DEFAULT_WORKBOOK)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output directory (default: <workbook_dir>/figures).",
    )
    parser.add_argument("--bins", type=int, default=5)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("png", "pdf", "svg"),
        default=["png"],
    )
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--estimators", nargs="+", default=None)
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        help=(
            "Metric names to plot (default: every metric listed in the "
            "workbook's Metric Definitions sheet)."
        ),
    )
    completeness_group = parser.add_mutually_exclusive_group()
    completeness_group.add_argument(
        "--include_incomplete",
        dest="include_incomplete",
        action="store_true",
        help="Include partial configurations (default).",
    )
    completeness_group.add_argument(
        "--complete_only",
        dest="include_incomplete",
        action="store_false",
        help="Plot only configurations with every expected rollout pair.",
    )
    parser.set_defaults(include_incomplete=True)
    parser.add_argument("--no_annotations", action="store_true")
    parser.add_argument("--log_x", action="store_true")
    return parser.parse_args(argv)


def run_analysis(args) -> Dict[str, Any]:
    workbook = args.workbook.resolve()
    if not workbook.is_file():
        raise FileNotFoundError(workbook)
    if args.bins < 2:
        raise ValueError("--bins must be at least 2")
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive")
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else workbook.parent / "figures"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_specs = load_metric_specs(workbook, requested=args.metrics)
    rollout_rows, expected_by_config = load_rollout_rows(
        workbook,
        tasks=set(args.tasks) if args.tasks else None,
        estimators=set(args.estimators) if args.estimators else None,
    )
    all_paths = []
    metric_payloads = {}
    selected_keys = set()
    complete_keys = set()
    for metric in metric_specs:
        summaries, grouped_rows = summarize_configurations(
            rollout_rows,
            expected_by_config,
            metric,
        )
        selected = [
            summary
            for summary in summaries
            if args.include_incomplete or summary.is_complete
        ]
        if not selected:
            completeness_hint = (
                "; use --include_incomplete to plot partial evaluations"
                if not args.include_incomplete else ""
            )
            raise RuntimeError(
                f"No matching configurations contain {metric.column!r}"
                f"{completeness_hint}. Regenerate the workbook with that metric."
            )

        global_paths, correlations = plot_global_metric_vs_success(
            selected,
            metric,
            output_dir,
            args.formats,
            args.dpi,
            annotate=not args.no_annotations,
            log_x=args.log_x,
        )
        task_paths = plot_task_success_curves(
            selected,
            grouped_rows,
            metric,
            output_dir,
            args.formats,
            args.dpi,
            bin_count=args.bins,
            log_x=args.log_x,
        )
        distribution_paths = plot_success_failure_distributions(
            selected,
            grouped_rows,
            metric,
            output_dir,
            args.formats,
            args.dpi,
            log_x=args.log_x,
        )
        metric_paths = global_paths + task_paths + distribution_paths
        all_paths.extend(metric_paths)
        selected_keys.update(summary.key for summary in selected)
        complete_keys.update(
            summary.key for summary in selected if summary.is_complete
        )
        metric_payloads[metric.name] = {
            "display_name": metric.display_name,
            "units": metric.units,
            "lower_is_better": metric.lower_is_better,
            "configuration_count": len(selected),
            "complete_configuration_count": sum(
                summary.is_complete for summary in selected
            ),
            "global_correlations": correlations,
            "figures": [str(path) for path in metric_paths],
        }

    selected_rows = [
        row for row in rollout_rows if _config_key(row) in selected_keys
    ]
    estimator_rows = build_estimator_summary_rows(selected_rows, metric_specs)
    estimator_table_path = output_dir / "estimator_sr_and_metrics.csv"
    write_estimator_summary_csv(
        estimator_table_path,
        estimator_rows,
        metric_specs,
    )
    payload = {
        "workbook": str(workbook),
        "output_dir": str(output_dir),
        "configuration_count": len(selected_keys),
        "complete_configuration_count": len(complete_keys),
        "rollout_count": len(selected_rows),
        "task_count": len({_text(row, "task_name") for row in selected_rows}),
        "estimator_count": len({
            _text(row, "depth_estimator") for row in selected_rows
        }),
        "metric_count": len(metric_specs),
        "metrics": metric_payloads,
        "estimator_table": str(estimator_table_path),
        "estimator_table_aggregation": (
            "Success rate and metric means are calculated across rollouts; "
            "each rollout has equal weight."
        ),
        "estimator_table_metric_units": {
            metric.name: metric.units for metric in metric_specs
        },
        "global_correlations": metric_payloads.get("chamfer", {}).get(
            "global_correlations",
            {},
        ),
        "metric_correlations": {
            name: result["global_correlations"]
            for name, result in metric_payloads.items()
        },
        "figures": [str(path) for path in all_paths],
    }
    summary_path = output_dir / "ip_analysis_figures_summary.json"
    write_json_atomic(summary_path, payload)
    payload["summary"] = str(summary_path)
    return payload


def main(argv=None):
    args = parse_args(argv)
    summary = run_analysis(args)
    print(
        f"Created {len(summary['figures'])} figure files from "
        f"{summary['rollout_count']} rollouts across "
        f"{summary['task_count']} tasks."
    )
    for path in summary["figures"]:
        print(path)
    print(summary["estimator_table"])
    print(summary["summary"])


if __name__ == "__main__":
    main()
