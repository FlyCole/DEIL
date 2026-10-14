#!/usr/bin/env python3
"""Create organized aggregate figures and tables from DP3 statistics.

The input directory is scanned recursively. Both structured artifact paths
(``task/estimator/cams-*/res-*/mask-*/best_stats.json``) and Hydra DP3 run
folders (``rlb_<task>-dp3-<tag>-<estimator>_seed<N>``) are supported. By
default, portable stats are loaded from ``data_analysis/dp3/stats`` and output
is grouped under ``data_analysis/dp3/outputs``.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from data_analysis.ip.plot_ip_rollout_analysis import (  # noqa: E402
    MetricSpec,
    plot_global_metric_vs_success,
    plot_success_failure_distributions,
    plot_task_success_curves,
    summarize_configurations,
    write_json_atomic,
)
from plot_cd_vs_success import plot_cd_vs_success_yerr  # noqa: E402


DEFAULT_INPUT_DIR = REPOSITORY_ROOT / "data_analysis" / "dp3" / "stats"
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "data_analysis" / "dp3" / "outputs"
KNOWN_METRIC_ORDER = (
    "chamfer",
    "accuracy_mm",
    "completeness_mm",
    "temporal_inconsistency_mm",
    "precision_5mm",
    "recall_5mm",
    "fscore_5mm",
    "precision_10mm",
    "recall_10mm",
    "fscore_10mm",
    "precision_20mm",
    "recall_20mm",
    "fscore_20mm",
)
DEFAULT_ANALYSIS_METRICS = (
    "chamfer",
    "accuracy_mm",
    "completeness_mm",
    "temporal_inconsistency_mm",
    "fscore_20mm",
)
HYDRA_RUN_PATTERN = re.compile(
    r"^rlb_(?P<task>.+)-dp3-(?P<tag>[^-]+)-"
    r"(?P<estimator>.+)_seed(?P<seed>\d+)$"
)
ROLLOUT_ID_PATTERN = re.compile(r"rollout_(\d+)")
TOTAL_METRIC_FILE_NAMES = {
    "chamfer": "cd_vs_success",
    "accuracy_mm": "accuracy_vs_success",
    "completeness_mm": "completeness_vs_success",
    "temporal_inconsistency_mm": "temporal_inconsistency_vs_success",
    "fscore_20mm": "fscore_20mm_vs_success",
}
ESTIMATOR_SUCCESS_FAILURE_METRICS = (
    "accuracy_mm",
    "completeness_mm",
    "fscore_20mm",
)
GEOMETRIC_PROFILE_COLUMNS = (
    ("success_rate", "SR ↑", False, ".3f"),
    ("chamfer_mean", "CD ↓", True, ".5f"),
    ("accuracy_mm_mean", "Accuracy (mm) ↓", True, ".2f"),
    (
        "completeness_mm_mean",
        "Completeness (mm) ↓",
        True,
        ".2f",
    ),
    ("fscore_20mm_mean", "F-score @ 20 mm ↑", False, ".3f"),
)
GEOMETRIC_PROFILE_HEATMAP_COLUMNS = tuple(
    column
    for column in GEOMETRIC_PROFILE_COLUMNS
    if column[0] != "chamfer_mean"
)
DEPTH_ESTIMATOR_PROFILE_ORDER = (
    "gt",
    "pi3",
    "vggt",
    "moge",
    "mast3r",
    "map_anything",
)
GT_NOISE_PROFILE_ORDER = (
    "gt",
    "gt_noisy_10mm",
    "gt_noisy_20mm",
    "gt_noisy_30mm",
    "gt_noisy_40mm",
)
COMBINED_GEOMETRIC_PROFILE_ORDER = (
    *GT_NOISE_PROFILE_ORDER,
    *DEPTH_ESTIMATOR_PROFILE_ORDER[1:],
)
GEOMETRIC_PROFILE_TICK_FONTSIZE = 13
GEOMETRIC_PROFILE_CELL_FONTSIZE = 13
GEOMETRIC_PROFILE_PANEL_TITLE_FONTSIZE = 16
GEOMETRIC_PROFILE_MAIN_TITLE_FONTSIZE = 18
GEOMETRIC_PROFILE_COLORBAR_TICK_FONTSIZE = 12
GEOMETRIC_PROFILE_COLORBAR_LABEL_FONTSIZE = 14
GEOMETRIC_PROFILE_FIGURE_WIDTH = 10.4


@dataclass(frozen=True)
class DP3Run:
    stats_path: Path
    run_name: str
    task_name: str
    estimator: str
    camera_names: Tuple[str, ...]
    resolution: int
    mask_input: str
    seed: int
    success_count: int
    rollout_count: int
    metrics: Dict[str, Dict[str, Any]]
    rollout_rows: Tuple[Dict[str, Any], ...]

    @property
    def success_rate(self) -> float:
        return self.success_count / self.rollout_count

    @property
    def config_key(self) -> Tuple[str, str, str, int, str, str]:
        return (
            self.task_name,
            self.estimator,
            "|".join(self.camera_names),
            self.resolution,
            self.mask_input,
            self.run_name,
        )


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().casefold()
    if text in {"true", "1", "yes", "success"}:
        return True
    if text in {"false", "0", "no", "failure"}:
        return False
    raise ValueError(f"Cannot interpret success value {value!r}")


def _metric_units(name: str) -> str:
    if name in {
        "accuracy_mm",
        "completeness_mm",
        "temporal_inconsistency_mm",
    }:
        return "mm"
    if name.startswith(("precision_", "recall_", "fscore_")):
        return "fraction"
    if name == "chamfer":
        return "normalized distance"
    return ""


def _lower_is_better(name: str) -> bool:
    return not name.startswith(("precision_", "recall_", "fscore_"))


def _structured_metadata(stats_path: Path) -> Dict[str, Any]:
    mask_dir = stats_path.parent
    if not mask_dir.name.startswith("mask-"):
        return {}
    resolution_dir = mask_dir.parent
    cameras_dir = resolution_dir.parent
    estimator_dir = cameras_dir.parent
    task_dir = estimator_dir.parent
    if (
        not resolution_dir.name.startswith("res-")
        or not cameras_dir.name.startswith("cams-")
    ):
        return {}
    try:
        resolution = int(resolution_dir.name.removeprefix("res-"))
    except ValueError:
        resolution = 0
    return {
        "task_name": task_dir.name,
        "estimator": estimator_dir.name,
        "camera_names": tuple(
            name for name in cameras_dir.name.removeprefix("cams-").split("+")
            if name
        ),
        "resolution": resolution,
        "mask_input": mask_dir.name.removeprefix("mask-"),
        "seed": 0,
    }


def _hydra_metadata(run_dir: Path) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    match = HYDRA_RUN_PATTERN.fullmatch(run_dir.name)
    if match is not None:
        metadata.update({
            "task_name": match.group("task"),
            "estimator": match.group("estimator"),
            "seed": int(match.group("seed")),
        })

    config_path = run_dir / ".hydra" / "config.yaml"
    if not config_path.is_file():
        return metadata
    try:
        with config_path.open(encoding="utf-8") as file:
            config = yaml.safe_load(file) or {}
    except (OSError, ValueError, TypeError, yaml.YAMLError):
        return metadata
    task_config = config.get("task") if isinstance(config, dict) else None
    if not isinstance(task_config, dict):
        return metadata
    runner = task_config.get("env_runner")
    runner = runner if isinstance(runner, dict) else {}
    metadata.update({
        "task_name": task_config.get(
            "task_name",
            metadata.get("task_name", "unknown_task"),
        ),
        "estimator": task_config.get(
            "depth_est_name",
            metadata.get("estimator", "unknown_estimator"),
        ),
        "camera_names": tuple(str(value) for value in runner.get(
            "camera_names",
            (),
        )),
        "resolution": int(runner.get("resolution", 0) or 0),
    })
    return metadata


def _run_metadata(stats_path: Path, stats: Dict[str, Any]) -> Dict[str, Any]:
    metadata = _structured_metadata(stats_path)
    if not metadata:
        metadata = _hydra_metadata(stats_path.parent)
    metadata.update({
        key: value
        for key, value in {
            "task_name": stats.get("task_name"),
            "estimator": stats.get("depth_estimator"),
            "camera_names": stats.get("camera_names"),
            "resolution": stats.get("resolution"),
            "mask_input": stats.get("mask_input"),
        }.items()
        if value is not None
    })
    cameras = metadata.get("camera_names", ())
    if isinstance(cameras, str):
        cameras = tuple(name for name in cameras.split("|") if name)
    metadata["camera_names"] = tuple(str(name) for name in cameras)
    metadata.setdefault("task_name", "unknown_task")
    metadata.setdefault("estimator", "unknown_estimator")
    metadata.setdefault("resolution", 0)
    metadata.setdefault("mask_input", "unknown")
    metadata.setdefault("seed", 0)
    return metadata


def _rollout_metric_summary(metric_payload: Dict[str, Any]) -> Dict[str, Any]:
    summary = metric_payload.get("summary")
    return summary if isinstance(summary, dict) else metric_payload


def load_dp3_run(stats_path: Path) -> DP3Run:
    with stats_path.open(encoding="utf-8") as file:
        stats = json.load(file)
    if "best_eval_dir" not in stats or not isinstance(stats.get("rollouts"), list):
        raise ValueError("not a process_dp3_rollout_artifacts best_stats file")

    metadata = _run_metadata(stats_path, stats)
    run_name = stats_path.parent.name
    rollout_rows = []
    for fallback_id, rollout in enumerate(stats["rollouts"]):
        source_npz = str(rollout.get("source_npz", ""))
        id_match = ROLLOUT_ID_PATTERN.search(Path(source_npz).stem)
        rollout_id = int(id_match.group(1)) if id_match else fallback_id
        success = _as_bool(rollout.get("success", rollout.get("outcome")))
        row = {
            "task_name": str(metadata["task_name"]),
            "depth_estimator": str(metadata["estimator"]),
            "camera_names": "|".join(metadata["camera_names"]),
            "resolution": int(metadata["resolution"]),
            "mask_input": str(metadata["mask_input"]),
            "eval_name": run_name,
            "rollout_id": rollout_id,
            "success": success,
            "source_npz": source_npz,
        }
        rollout_metrics = rollout.get("metrics")
        if not isinstance(rollout_metrics, dict):
            rollout_metrics = {}
        for metric_name, metric_payload in rollout_metrics.items():
            if not isinstance(metric_payload, dict):
                continue
            summary = _rollout_metric_summary(metric_payload)
            try:
                value = float(summary["mean"])
            except (KeyError, TypeError, ValueError):
                continue
            if np.isfinite(value):
                row[f"{metric_name}_mean"] = value
        rollout_rows.append(row)

    if not rollout_rows:
        raise ValueError("best_stats contains no rollout metric records")
    success_count = sum(row["success"] for row in rollout_rows)
    return DP3Run(
        stats_path=stats_path,
        run_name=run_name,
        task_name=str(metadata["task_name"]),
        estimator=str(metadata["estimator"]),
        camera_names=metadata["camera_names"],
        resolution=int(metadata["resolution"]),
        mask_input=str(metadata["mask_input"]),
        seed=int(metadata["seed"]),
        success_count=int(success_count),
        rollout_count=len(rollout_rows),
        metrics=stats.get("metrics") or {},
        rollout_rows=tuple(rollout_rows),
    )


def discover_runs(
    input_dir: Path,
    tasks: Optional[set[str]] = None,
    estimators: Optional[set[str]] = None,
) -> Tuple[list[DP3Run], list[str]]:
    runs = []
    skipped = []
    for stats_path in sorted(input_dir.rglob("best_stats.json")):
        try:
            run = load_dp3_run(stats_path)
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            skipped.append(f"{stats_path}: {type(exc).__name__}: {exc}")
            continue
        if tasks is not None and run.task_name not in tasks:
            continue
        if estimators is not None and run.estimator not in estimators:
            continue
        runs.append(run)
    return runs, skipped


def build_metric_specs(
    runs: Sequence[DP3Run],
    requested: Optional[Sequence[str]],
    allow_missing: bool = False,
) -> list[MetricSpec]:
    available = {
        name
        for run in runs
        for row in run.rollout_rows
        for name in KNOWN_METRIC_ORDER
        if f"{name}_mean" in row
    }
    extra_names = sorted({
        key.removesuffix("_mean")
        for run in runs
        for row in run.rollout_rows
        for key in row
        if key.endswith("_mean") and key.removesuffix("_mean") not in available
    })
    ordered = [name for name in KNOWN_METRIC_ORDER if name in available]
    ordered.extend(extra_names)
    names = list(requested) if requested else ordered
    missing = [name for name in names if name not in set(ordered)]
    if missing and not allow_missing:
        raise ValueError(
            f"Requested metric(s) {missing} are unavailable; found {ordered}"
        )
    if allow_missing:
        names = [name for name in names if name in set(ordered)]

    definitions = {}
    for run in runs:
        for name, definition in run.metrics.items():
            if isinstance(definition, dict):
                definitions.setdefault(name, definition)
    return [
        MetricSpec(
            name=name,
            display_name=str(
                definitions.get(name, {}).get("display_name")
                or name.replace("_", " ").title()
            ),
            units=str(
                definitions.get(name, {}).get("units")
                or _metric_units(name)
            ),
            lower_is_better=_lower_is_better(name),
        )
        for name in names
    ]


def _run_metric_mean(run: DP3Run, metric: MetricSpec) -> Optional[float]:
    values = [
        float(row[metric.column])
        for row in run.rollout_rows
        if metric.column in row
    ]
    return float(np.mean(values)) if values else None


def _legacy_estimator_label(estimator: str) -> str:
    normalized = estimator.casefold()
    noisy_match = re.fullmatch(r"gt_noisy_(\d+)mm", normalized)
    if noisy_match is not None:
        return f"Noisy_{noisy_match.group(1)}mm"
    labels = {
        "gt": "Depth",
        "pi3": "Pi3",
        "vggt": "VGGT",
        "moge": "MoGe",
        "mast3r": "MASt3R",
        "map_anything": "MapAnything",
    }
    return labels.get(normalized, estimator)


def _profile_estimator_label(estimator: str) -> str:
    normalized = estimator.casefold()
    if normalized == "gt":
        return "GT"
    noisy_match = re.fullmatch(r"gt_noisy_(\d+)mm", normalized)
    if noisy_match is not None:
        return f"GT Noisy {noisy_match.group(1)} mm"
    return _legacy_estimator_label(estimator)


def _geometric_profile_estimator_label(estimator: str) -> str:
    normalized = estimator.casefold()
    if normalized == "gt":
        return "Depth"
    noisy_match = re.fullmatch(r"gt_noisy_(\d+)mm", normalized)
    if noisy_match is not None:
        return f"Noisy_{noisy_match.group(1)}mm"
    return _profile_estimator_label(estimator)


def _estimator_sort_key(estimator: str) -> Tuple[int, str]:
    normalized = estimator.casefold()
    try:
        order_index = COMBINED_GEOMETRIC_PROFILE_ORDER.index(normalized)
    except ValueError:
        order_index = len(COMBINED_GEOMETRIC_PROFILE_ORDER)
    return order_index, _profile_estimator_label(estimator).casefold()


def _geometric_profile_sort_key(profile: Dict[str, Any]) -> Tuple[int, str]:
    return _estimator_sort_key(str(profile["depth_estimator"]))


def select_geometric_profile_group(
    profiles: Sequence[Dict[str, Any]],
    estimator_order: Sequence[str],
) -> list[Dict[str, Any]]:
    """Select available estimator profiles in an explicit display order."""
    profiles_by_estimator = {
        str(profile["depth_estimator"]).casefold(): profile
        for profile in profiles
    }
    return [
        profiles_by_estimator[estimator.casefold()]
        for estimator in estimator_order
        if estimator.casefold() in profiles_by_estimator
    ]


def total_metric_pairs(
    runs: Sequence[DP3Run],
    metric: MetricSpec,
) -> list[Tuple[str, str, float, float, float]]:
    grouped: Dict[Tuple[str, str], list[DP3Run]] = {}
    for run in runs:
        grouped.setdefault((run.estimator, run.task_name), []).append(run)

    estimator_task_values: Dict[str, list[Tuple[float, float]]] = {}
    for (estimator, _), task_runs in grouped.items():
        metric_values = [
            float(row[metric.column])
            for run in task_runs
            for row in run.rollout_rows
            if metric.column in row
        ]
        success_values = [
            float(row["success"])
            for run in task_runs
            for row in run.rollout_rows
        ]
        if metric_values and success_values:
            estimator_task_values.setdefault(estimator, []).append((
                float(np.mean(metric_values)),
                float(np.mean(success_values)),
            ))

    pairs = []
    for estimator, task_values in sorted(estimator_task_values.items()):
        metric_values = np.asarray([value for value, _ in task_values])
        success_values = np.asarray([value for _, value in task_values])
        pairs.append((
            _legacy_estimator_label(estimator),
            f"point_cloud_{estimator}",
            float(metric_values.mean()),
            float(success_values.mean()),
            float(success_values.std()),
        ))
    return pairs


def legacy_cd_pairs(
    runs: Sequence[DP3Run],
) -> list[Tuple[str, str, float, float, float]]:
    """Backward-compatible helper for callers that only request Chamfer."""
    chamfer = MetricSpec("chamfer", "Chamfer", "normalized distance", True)
    return total_metric_pairs(runs, chamfer)


def plot_accuracy_temporal_tradeoff(
    estimator_rows: Sequence[Dict[str, Any]],
    output_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> Tuple[list[Path], Optional[str]]:
    """Plot task-balanced accuracy and temporal stability for each method."""
    selected_rows = []
    for row in estimator_rows:
        try:
            accuracy = float(row["accuracy_mm_mean"])
            temporal = float(row["temporal_inconsistency_mm_mean"])
            success_rate = float(row["success_rate"])
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite((accuracy, temporal, success_rate)).all():
            selected_rows.append((row, accuracy, temporal, success_rate))
    if len(selected_rows) < 2:
        return [], (
            "Skipped accuracy-vs-temporal-inconsistency because at least "
            "two estimators with both metrics are required."
        )

    selected_rows.sort(
        key=lambda item: _estimator_sort_key(
            str(item[0]["depth_estimator"])
        )
    )
    figure, axis = plt.subplots(figsize=(8.2, 5.8), constrained_layout=True)
    accuracies = np.asarray([item[1] for item in selected_rows])
    temporal_values = np.asarray([item[2] for item in selected_rows])
    success_rates = np.asarray([item[3] for item in selected_rows])
    points = axis.scatter(
        accuracies,
        temporal_values,
        c=success_rates,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        s=85,
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )
    for row, accuracy, temporal, _ in selected_rows:
        axis.annotate(
            _legacy_estimator_label(str(row["depth_estimator"])),
            (accuracy, temporal),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
        )
    axis.set_xlabel("Average accuracy distance (mm; lower is better)")
    axis.set_ylabel(
        "GT-compensated temporal inconsistency (mm; lower is better)"
    )
    axis.set_title("Accuracy vs Temporal Consistency for Each Method")
    axis.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    color_bar = figure.colorbar(points, ax=axis, pad=0.02)
    color_bar.set_label("Average success rate")
    color_bar.ax.yaxis.set_major_formatter(PercentFormatter(1.0))

    paths = []
    for extension in formats:
        path = output_dir / (
            f"accuracy_vs_temporal_inconsistency.{extension}"
        )
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
        paths.append(path)
    plt.close(figure)
    return paths, None


def write_csv_atomic(
    path: Path,
    rows: Sequence[Dict[str, Any]],
    headers: Sequence[str],
) -> None:
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


def remove_obsolete_metric_outputs(
    output_dir: Path,
    selected_metric_names: set[str],
) -> list[Path]:
    """Remove known generated plots for metrics excluded from this analysis."""
    removed = []
    obsolete_metrics = set(KNOWN_METRIC_ORDER) - selected_metric_names
    for metric_name in sorted(obsolete_metrics):
        stems = (
            f"{metric_name}_vs_success_rate",
            f"task_{metric_name}_vs_success_rate",
            f"success_failure_{metric_name}_by_task",
        )
        metric_dir = output_dir / "task_level" / metric_name
        for stem in stems:
            for extension in ("png", "pdf", "svg"):
                path = metric_dir / f"{stem}.{extension}"
                if path.is_file():
                    path.unlink()
                    removed.append(path)
        try:
            metric_dir.rmdir()
        except OSError:
            pass

    # Remove figures from the previous pre-level output layout. Only known
    # generated filenames are touched; unrelated files are preserved.
    legacy_metric_root = output_dir / "metrics"
    for metric_name in KNOWN_METRIC_ORDER:
        metric_dir = legacy_metric_root / metric_name
        stems = (
            f"{metric_name}_vs_success_rate",
            f"task_{metric_name}_vs_success_rate",
            f"success_failure_{metric_name}_by_task",
        )
        for stem in stems:
            for extension in ("png", "pdf", "svg"):
                path = metric_dir / f"{stem}.{extension}"
                if path.is_file():
                    path.unlink()
                    removed.append(path)
        try:
            metric_dir.rmdir()
        except OSError:
            pass
    try:
        legacy_metric_root.rmdir()
    except OSError:
        pass

    obsolete_total_stems = {
        "fscore_5mm": "fscore_5mm_vs_success",
        "fscore_10mm": "fscore_10mm_vs_success",
        "temporal_inconsistency_mm": "temporal_inconsistency_vs_success",
    }
    for metric_name, stem in obsolete_total_stems.items():
        if metric_name in selected_metric_names:
            continue
        for extension in ("png", "pdf", "svg"):
            path = output_dir / "estimator_level" / f"{stem}.{extension}"
            if path.is_file():
                path.unlink()
                removed.append(path)
    if "temporal_inconsistency_mm" not in selected_metric_names:
        for extension in ("png", "pdf", "svg"):
            path = (
                output_dir
                / "estimator_level"
                / f"accuracy_vs_temporal_inconsistency.{extension}"
            )
            if path.is_file():
                path.unlink()
                removed.append(path)

    legacy_total_dir = output_dir / "total"
    for stem in (
        "cd_vs_success",
        "accuracy_vs_success",
        "completeness_vs_success",
        "fscore_5mm_vs_success",
        "fscore_10mm_vs_success",
        "fscore_20mm_vs_success",
        "temporal_inconsistency_vs_success",
        "accuracy_vs_temporal_inconsistency",
    ):
        for extension in ("png", "pdf", "svg"):
            path = legacy_total_dir / f"{stem}.{extension}"
            if path.is_file():
                path.unlink()
                removed.append(path)
    try:
        legacy_total_dir.rmdir()
    except OSError:
        pass

    for relative_path in (
        Path("tables/dp3_estimator_sr_and_metrics.csv"),
        Path("tables/dp3_run_summary.csv"),
        Path("summaries/dp3_analysis_summary.json"),
    ):
        path = output_dir / relative_path
        if path.is_file():
            path.unlink()
            removed.append(path)
    for legacy_dir_name in ("tables", "summaries"):
        try:
            (output_dir / legacy_dir_name).rmdir()
        except OSError:
            pass
    return removed


def build_estimator_geometric_profiles(
    rollout_rows: Sequence[Dict[str, Any]],
) -> list[Dict[str, Any]]:
    """Average rollouts within task, then average tasks equally by estimator."""
    grouped: Dict[Tuple[str, str], list[Dict[str, Any]]] = {}
    for row in rollout_rows:
        estimator = str(row.get("depth_estimator", "unknown_estimator"))
        task = str(row.get("task_name", "unknown_task"))
        grouped.setdefault((estimator, task), []).append(row)

    task_profiles: Dict[str, list[Dict[str, Any]]] = {}
    for (estimator, task), rows in grouped.items():
        profile: Dict[str, Any] = {"task_name": task}
        for column, _, _, _ in GEOMETRIC_PROFILE_COLUMNS:
            source_column = "success" if column == "success_rate" else column
            values = []
            for row in rows:
                try:
                    value = float(row[source_column])
                except (KeyError, TypeError, ValueError):
                    continue
                if np.isfinite(value):
                    values.append(value)
            if values:
                profile[column] = float(np.mean(values))
        task_profiles.setdefault(estimator, []).append(profile)

    estimator_profiles = []
    for estimator, profiles in task_profiles.items():
        estimator_profile: Dict[str, Any] = {
            "depth_estimator": estimator,
            "display_name": _geometric_profile_estimator_label(estimator),
            "task_count": len(profiles),
        }
        for column, _, _, _ in GEOMETRIC_PROFILE_COLUMNS:
            task_values = [
                float(profile[column])
                for profile in profiles
                if column in profile
            ]
            estimator_profile[f"{column}_task_count"] = len(task_values)
            estimator_profile[column] = (
                float(np.mean(task_values)) if task_values else None
            )
        estimator_profiles.append(estimator_profile)
    return sorted(
        estimator_profiles,
        key=_geometric_profile_sort_key,
    )


def write_estimator_geometric_profiles(
    path: Path,
    profiles: Sequence[Dict[str, Any]],
) -> None:
    headers = ["depth_estimator", "display_name", "task_count"]
    for column, _, _, _ in GEOMETRIC_PROFILE_COLUMNS:
        headers.extend((column, f"{column}_task_count"))
    write_csv_atomic(path, profiles, headers)


def _geometric_profile_heatmap_matrices(
    profiles: Sequence[Dict[str, Any]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Return raw annotations and column-normalized quality colors."""
    if not profiles:
        raise ValueError("No estimator profiles are available for the heatmap")
    raw_values = np.full(
        (len(profiles), len(GEOMETRIC_PROFILE_HEATMAP_COLUMNS)),
        np.nan,
        dtype=np.float64,
    )
    for row_index, profile in enumerate(profiles):
        for column_index, (column, _, _, _) in enumerate(
            GEOMETRIC_PROFILE_HEATMAP_COLUMNS
        ):
            value = profile.get(column)
            if value is not None and np.isfinite(float(value)):
                raw_values[row_index, column_index] = float(value)

    missing_columns = [
        label
        for column_index, (_, label, _, _) in enumerate(
            GEOMETRIC_PROFILE_HEATMAP_COLUMNS
        )
        if not np.isfinite(raw_values[:, column_index]).any()
    ]
    if missing_columns:
        raise ValueError(
            "Cannot create geometric-profile heatmap; unavailable columns: "
            + ", ".join(missing_columns)
        )

    quality_values = np.full_like(raw_values, np.nan)
    for column_index, (_, _, lower_is_better, _) in enumerate(
        GEOMETRIC_PROFILE_HEATMAP_COLUMNS
    ):
        column_values = raw_values[:, column_index]
        finite = np.isfinite(column_values)
        minimum = float(column_values[finite].min())
        maximum = float(column_values[finite].max())
        if np.isclose(minimum, maximum):
            normalized = np.full(finite.sum(), 0.5)
        else:
            normalized = (
                column_values[finite] - minimum
            ) / (maximum - minimum)
        if lower_is_better:
            normalized = 1.0 - normalized
        quality_values[finite, column_index] = normalized
    return raw_values, quality_values


def _draw_geometric_profile_heatmap_axis(
    axis,
    profiles: Sequence[Dict[str, Any]],
    raw_values: np.ndarray,
    quality_values: np.ndarray,
    title: str,
    divider_after_estimator: Optional[str],
):
    """Draw one geometric-profile panel from pre-normalized colors."""

    color_map = plt.get_cmap("YlGn").copy()
    color_map.set_bad("#E5E7EB")
    image = axis.imshow(
        np.ma.masked_invalid(quality_values),
        cmap=color_map,
        vmin=0.0,
        vmax=1.0,
        aspect="auto",
    )
    axis.set_xticks(
        np.arange(len(GEOMETRIC_PROFILE_HEATMAP_COLUMNS)),
        [column[1] for column in GEOMETRIC_PROFILE_HEATMAP_COLUMNS],
    )
    axis.set_yticks(
        np.arange(len(profiles)),
        [profile["display_name"] for profile in profiles],
    )
    axis.tick_params(
        axis="x",
        top=True,
        bottom=False,
        labeltop=True,
        labelbottom=False,
        labelsize=GEOMETRIC_PROFILE_TICK_FONTSIZE,
    )
    axis.tick_params(
        axis="y",
        labelsize=GEOMETRIC_PROFILE_TICK_FONTSIZE,
    )
    axis.set_xticks(
        np.arange(-0.5, len(GEOMETRIC_PROFILE_HEATMAP_COLUMNS), 1),
        minor=True,
    )
    axis.set_yticks(np.arange(-0.5, len(profiles), 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=2.0)
    axis.tick_params(which="minor", bottom=False, left=False)
    if divider_after_estimator is not None:
        divider_name = divider_after_estimator.casefold()
        divider_indices = [
            row_index
            for row_index, profile in enumerate(profiles)
            if str(profile["depth_estimator"]).casefold() == divider_name
        ]
        if divider_indices:
            axis.axhline(
                divider_indices[0] + 0.5,
                color="#111827",
                linewidth=3.0,
                zorder=5,
            )

    for row_index in range(len(profiles)):
        for column_index, (_, _, _, number_format) in enumerate(
            GEOMETRIC_PROFILE_HEATMAP_COLUMNS
        ):
            raw_value = raw_values[row_index, column_index]
            quality = quality_values[row_index, column_index]
            annotation = (
                format(raw_value, number_format)
                if np.isfinite(raw_value)
                else "—"
            )
            text_color = (
                "white"
                if np.isfinite(quality) and quality > 0.62
                else "#111827"
            )
            axis.text(
                column_index,
                row_index,
                annotation,
                ha="center",
                va="center",
                color=text_color,
                fontsize=GEOMETRIC_PROFILE_CELL_FONTSIZE,
                fontweight="semibold",
            )

    axis.set_title(
        title,
        pad=6,
        fontsize=GEOMETRIC_PROFILE_PANEL_TITLE_FONTSIZE,
        fontweight="semibold",
    )
    return image


def plot_estimator_geometric_profile_heatmap(
    profiles: Sequence[Dict[str, Any]],
    output_dir: Path,
    formats: Sequence[str],
    dpi: int,
    *,
    file_stem: str = "geometric_profile_heatmap",
    title: str = "Estimator-Level Geometric Profile",
    divider_after_estimator: Optional[str] = "gt_noisy_40mm",
) -> list[Path]:
    """Plot raw estimator means with column-normalized quality colors."""
    raw_values, quality_values = _geometric_profile_heatmap_matrices(profiles)
    figure, axis = plt.subplots(
        figsize=(
            GEOMETRIC_PROFILE_FIGURE_WIDTH,
            max(2, 0.62 * len(profiles) + 2.3),
        ),
        constrained_layout=True,
    )
    image = _draw_geometric_profile_heatmap_axis(
        axis,
        profiles,
        raw_values,
        quality_values,
        title,
        divider_after_estimator,
    )
    color_bar = figure.colorbar(image, ax=axis, shrink=0.82, pad=0.02)
    color_bar.set_label(
        "Relative quality (normalised within each column; higher is better)",
        fontsize=GEOMETRIC_PROFILE_COLORBAR_LABEL_FONTSIZE,
    )
    color_bar.ax.tick_params(
        labelsize=GEOMETRIC_PROFILE_COLORBAR_TICK_FONTSIZE
    )

    paths = []
    for extension in formats:
        path = output_dir / f"{file_stem}.{extension}"
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
        paths.append(path)
    plt.close(figure)
    return paths


def plot_split_geometric_profile_combined_figure(
    depth_estimator_profiles: Sequence[Dict[str, Any]],
    gt_noise_profiles: Sequence[Dict[str, Any]],
    output_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> list[Path]:
    """Stack independently normalized estimator and corrupted-depth heatmaps."""
    depth_raw, depth_quality = _geometric_profile_heatmap_matrices(
        depth_estimator_profiles
    )
    noise_raw, noise_quality = _geometric_profile_heatmap_matrices(
        gt_noise_profiles
    )
    panel_heights = (
        max(2.6, 0.62 * len(depth_estimator_profiles) + 1.2),
        max(2.6, 0.62 * len(gt_noise_profiles) + 1.2),
    )
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(GEOMETRIC_PROFILE_FIGURE_WIDTH, sum(panel_heights)),
        gridspec_kw={"height_ratios": panel_heights},
        constrained_layout=True,
    )
    depth_image = _draw_geometric_profile_heatmap_axis(
        axes[0],
        depth_estimator_profiles,
        depth_raw,
        depth_quality,
        "Depth Estimators",
        divider_after_estimator=None,
    )
    _draw_geometric_profile_heatmap_axis(
        axes[1],
        gt_noise_profiles,
        noise_raw,
        noise_quality,
        "Corrupted-Depth Baselines",
        divider_after_estimator=None,
    )
    figure.suptitle(
        "Geometric Profile Heatmap (n = 10 tasks)",
        fontsize=GEOMETRIC_PROFILE_MAIN_TITLE_FONTSIZE,
        fontweight="semibold",
    )
    color_bar = figure.colorbar(
        depth_image,
        ax=axes,
        shrink=0.72,
        pad=0.02,
    )
    color_bar.set_label(
        "Relative quality (normalised within each panel and column; "
        "higher is better)",
        fontsize=GEOMETRIC_PROFILE_COLORBAR_LABEL_FONTSIZE,
    )
    color_bar.ax.tick_params(
        labelsize=GEOMETRIC_PROFILE_COLORBAR_TICK_FONTSIZE
    )

    paths = []
    for extension in formats:
        path = output_dir / f"geometric_profile_heatmap_split_combined.{extension}"
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
        paths.append(path)
    plt.close(figure)
    return paths


def task_normalized_outcome_values(
    rollout_rows: Sequence[Dict[str, Any]],
    metric_specs: Sequence[MetricSpec],
) -> Dict[str, Dict[str, Any]]:
    """Normalize jointly within task, then split rollout z-scores by outcome."""
    output: Dict[str, Dict[str, Any]] = {}
    tasks = sorted({str(row["task_name"]) for row in rollout_rows})
    for metric in metric_specs:
        successful = []
        failed = []
        records = []
        included_tasks = []
        excluded_tasks = []
        for task in tasks:
            task_rows = [
                row
                for row in rollout_rows
                if str(row["task_name"]) == task and metric.column in row
            ]
            values = np.asarray(
                [float(row[metric.column]) for row in task_rows],
                dtype=np.float64,
            )
            if len(values) < 2 or not np.isfinite(values).all():
                excluded_tasks.append(task)
                continue
            task_std = float(values.std(ddof=0))
            if not np.isfinite(task_std) or np.isclose(task_std, 0.0):
                excluded_tasks.append(task)
                continue
            z_scores = (values - float(values.mean())) / task_std
            if not metric.lower_is_better:
                z_scores = -z_scores
            for row, z_score in zip(task_rows, z_scores):
                target = successful if row["success"] else failed
                target.append(float(z_score))
                records.append({
                    "task_name": task,
                    "depth_estimator": str(row["depth_estimator"]),
                    "success": bool(row["success"]),
                    "z_score": float(z_score),
                })
            included_tasks.append(task)
        output[metric.name] = {
            "success": np.asarray(successful, dtype=np.float64),
            "failure": np.asarray(failed, dtype=np.float64),
            "records": records,
            "included_tasks": included_tasks,
            "excluded_tasks": excluded_tasks,
            "sign_reversed": not metric.lower_is_better,
        }
    return output


def bootstrap_mean_confidence_interval(
    values: np.ndarray,
    sample_count: int,
    random: np.random.Generator,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Return rollout-bootstrap mean and percentile 95% confidence interval."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return None, None, None
    mean = float(values.mean())
    if len(values) == 1:
        return mean, mean, mean

    bootstrap_means = np.empty(sample_count, dtype=np.float64)
    batch_size = 256
    for start in range(0, sample_count, batch_size):
        stop = min(start + batch_size, sample_count)
        indices = random.integers(
            0,
            len(values),
            size=(stop - start, len(values)),
        )
        bootstrap_means[start:stop] = values[indices].mean(axis=1)
    lower, upper = np.quantile(bootstrap_means, (0.025, 0.975))
    return mean, float(lower), float(upper)


def summarize_task_normalized_outcomes(
    normalized: Dict[str, Dict[str, Any]],
    metric_specs: Sequence[MetricSpec],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> list[Dict[str, Any]]:
    random = np.random.default_rng(bootstrap_seed)
    rows = []
    for metric in metric_specs:
        metric_values = normalized[metric.name]
        for outcome in ("success", "failure"):
            values = metric_values[outcome]
            mean, lower, upper = bootstrap_mean_confidence_interval(
                values,
                bootstrap_samples,
                random,
            )
            rows.append({
                "metric": metric.name,
                "display_name": metric.display_name,
                "units": metric.units,
                "outcome": outcome,
                "rollout_count": len(values),
                "task_count": len(metric_values["included_tasks"]),
                "excluded_nonstandardizable_task_count": len(
                    metric_values["excluded_tasks"]
                ),
                "mean_z": mean,
                "bootstrap_ci95_lower": lower,
                "bootstrap_ci95_upper": upper,
                "bootstrap_samples": bootstrap_samples,
                "bootstrap_seed": bootstrap_seed,
                "higher_metric_is_better_sign_reversed": metric_values[
                    "sign_reversed"
                ],
            })
    return rows


def write_task_normalized_outcome_table(
    path: Path,
    rows: Sequence[Dict[str, Any]],
) -> None:
    headers = [
        "metric",
        "display_name",
        "units",
        "outcome",
        "rollout_count",
        "task_count",
        "excluded_nonstandardizable_task_count",
        "mean_z",
        "bootstrap_ci95_lower",
        "bootstrap_ci95_upper",
        "bootstrap_samples",
        "bootstrap_seed",
        "higher_metric_is_better_sign_reversed",
    ]
    write_csv_atomic(path, rows, headers)


def plot_task_normalized_success_failure(
    normalized: Dict[str, Dict[str, Any]],
    summary_rows: Sequence[Dict[str, Any]],
    metric_specs: Sequence[MetricSpec],
    output_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> list[Path]:
    """Plot pooled rollout distributions after joint task-wise z-scoring."""
    column_count = min(2, len(metric_specs))
    row_count = int(np.ceil(len(metric_specs) / column_count))
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(5.2 * column_count, 4.8 * row_count),
        squeeze=False,
        constrained_layout=True,
    )
    colors = {"success": "#4C9F70", "failure": "#D95F5F"}
    labels = {"success": "Successful", "failure": "Failed"}
    summary_lookup = {
        (str(row["metric"]), str(row["outcome"])): row
        for row in summary_rows
    }

    for metric_index, metric in enumerate(metric_specs):
        axis = axes.flat[metric_index]
        metric_values = normalized[metric.name]
        tick_labels = []
        for position, outcome in enumerate(("success", "failure")):
            values = metric_values[outcome]
            statistics = summary_lookup[(metric.name, outcome)]
            mean = statistics["mean_z"]
            lower = statistics["bootstrap_ci95_lower"]
            upper = statistics["bootstrap_ci95_upper"]
            if len(values):
                violin = axis.violinplot(
                    [values],
                    positions=[position],
                    widths=0.72,
                    showmeans=False,
                    showmedians=False,
                    showextrema=False,
                )
                for body in violin["bodies"]:
                    body.set_facecolor(colors[outcome])
                    body.set_edgecolor(colors[outcome])
                    body.set_alpha(0.48)
                box = axis.boxplot(
                    [values],
                    positions=[position],
                    widths=0.18,
                    patch_artist=True,
                    showfliers=False,
                    manage_ticks=False,
                )
                for patch in box["boxes"]:
                    patch.set_facecolor(colors[outcome])
                    patch.set_alpha(0.82)
                lower_error = max(0.0, mean - lower)
                upper_error = max(0.0, upper - mean)
                axis.errorbar(
                    [position],
                    [mean],
                    yerr=np.asarray([[lower_error], [upper_error]]),
                    fmt="D",
                    color="#111827",
                    markerfacecolor="white",
                    markersize=5,
                    linewidth=1.4,
                    capsize=3,
                    zorder=5,
                )
                tick_labels.append(
                    f"{labels[outcome]}\n"
                    f"n={len(values)}  μ={mean:.3f}\n"
                    f"95% CI [{lower:.3f}, {upper:.3f}]"
                )
            else:
                tick_labels.append(f"{labels[outcome]}\nn=0")
        axis.axhline(0.0, color="#4B5563", linestyle="--", linewidth=1.0)
        axis.set_xticks((0, 1), tick_labels)
        axis.set_xlim(-0.65, 1.65)
        axis.set_title(metric.display_name)
        axis.grid(True, axis="y", alpha=0.22)

    for empty_index in range(len(metric_specs), row_count * column_count):
        figure.delaxes(axes.flat[empty_index])
    figure.suptitle(
        "Successful vs. Failed Rollout Geometry\n"
        "Task-wise z-scores use both outcome groups and all estimators together",
        fontsize=15,
    )
    figure.supylabel("Task-normalised geometric error (z-score)")
    figure.supxlabel(
        "Positive values indicate worse-than-average geometry within the same task. "
        "Diamonds and error bars show group mean with bootstrap 95% CI.",
        fontsize=10,
    )

    paths = []
    for extension in formats:
        path = output_dir / f"task_normalised_success_vs_failure.{extension}"
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
        paths.append(path)
    plt.close(figure)
    return paths


def summarize_estimator_normalized_outcomes(
    normalized: Dict[str, Dict[str, Any]],
    metric_specs: Sequence[MetricSpec],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> list[Dict[str, Any]]:
    """Summarize task-normalized rollout z-scores by estimator and outcome."""
    random = np.random.default_rng(bootstrap_seed)
    rows = []
    for metric in metric_specs:
        records = normalized[metric.name]["records"]
        estimators = sorted(
            {
                str(record["depth_estimator"])
                for record in records
            },
            key=_estimator_sort_key,
        )
        for estimator in estimators:
            estimator_records = [
                record
                for record in records
                if record["depth_estimator"] == estimator
            ]
            for outcome, success in (("success", True), ("failure", False)):
                selected = [
                    record
                    for record in estimator_records
                    if record["success"] is success
                ]
                values = np.asarray(
                    [record["z_score"] for record in selected],
                    dtype=np.float64,
                )
                mean, lower, upper = bootstrap_mean_confidence_interval(
                    values,
                    bootstrap_samples,
                    random,
                )
                rows.append({
                    "metric": metric.name,
                    "display_name": metric.display_name,
                    "depth_estimator": estimator,
                    "estimator_display_name": _profile_estimator_label(
                        estimator
                    ),
                    "outcome": outcome,
                    "rollout_count": len(values),
                    "task_count": len({
                        str(record["task_name"]) for record in selected
                    }),
                    "mean_z": mean,
                    "bootstrap_ci95_lower": lower,
                    "bootstrap_ci95_upper": upper,
                    "bootstrap_samples": bootstrap_samples,
                    "bootstrap_seed": bootstrap_seed,
                })
    return rows


def write_estimator_normalized_outcome_table(
    path: Path,
    rows: Sequence[Dict[str, Any]],
) -> None:
    headers = [
        "metric",
        "display_name",
        "depth_estimator",
        "estimator_display_name",
        "outcome",
        "rollout_count",
        "task_count",
        "mean_z",
        "bootstrap_ci95_lower",
        "bootstrap_ci95_upper",
        "bootstrap_samples",
        "bootstrap_seed",
    ]
    write_csv_atomic(path, rows, headers)


def plot_estimator_normalized_success_failure(
    normalized: Dict[str, Dict[str, Any]],
    summary_rows: Sequence[Dict[str, Any]],
    metric_specs: Sequence[MetricSpec],
    output_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> list[Path]:
    """Compare successful and failed rollout geometry for each estimator."""
    column_count = len(metric_specs)
    row_count = 1
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(6.2 * column_count, 5.8),
        squeeze=False,
        constrained_layout=True,
    )
    colors = {"success": "#4C9F70", "failure": "#D95F5F"}
    offsets = {"success": -0.19, "failure": 0.19}
    summary_lookup = {
        (
            str(row["metric"]),
            str(row["depth_estimator"]),
            str(row["outcome"]),
        ): row
        for row in summary_rows
    }

    for metric_index, metric in enumerate(metric_specs):
        axis = axes.flat[metric_index]
        estimators = sorted(
            {
                str(record["depth_estimator"])
                for record in normalized[metric.name]["records"]
            },
            key=_estimator_sort_key,
        )
        for estimator_index, estimator in enumerate(estimators):
            records = [
                record
                for record in normalized[metric.name]["records"]
                if record["depth_estimator"] == estimator
            ]
            for outcome, success in (("success", True), ("failure", False)):
                values = np.asarray([
                    record["z_score"]
                    for record in records
                    if record["success"] is success
                ], dtype=np.float64)
                if len(values) == 0:
                    continue
                position = estimator_index + offsets[outcome]
                if len(values) >= 2 and not np.isclose(values.std(), 0.0):
                    violin = axis.violinplot(
                        [values],
                        positions=[position],
                        widths=0.34,
                        showmeans=False,
                        showmedians=False,
                        showextrema=False,
                    )
                    for body in violin["bodies"]:
                        body.set_facecolor(colors[outcome])
                        body.set_edgecolor(colors[outcome])
                        body.set_alpha(0.43)
                box = axis.boxplot(
                    [values],
                    positions=[position],
                    widths=0.11,
                    patch_artist=True,
                    showfliers=False,
                    manage_ticks=False,
                )
                for patch in box["boxes"]:
                    patch.set_facecolor(colors[outcome])
                    patch.set_alpha(0.82)

                statistics = summary_lookup[
                    (metric.name, estimator, outcome)
                ]
                mean = statistics["mean_z"]
                lower = statistics["bootstrap_ci95_lower"]
                upper = statistics["bootstrap_ci95_upper"]
                lower_error = max(0.0, mean - lower)
                upper_error = max(0.0, upper - mean)
                axis.errorbar(
                    [position],
                    [mean],
                    yerr=np.asarray([[lower_error], [upper_error]]),
                    fmt="D",
                    color="#111827",
                    markerfacecolor="white",
                    markersize=4,
                    linewidth=1.2,
                    capsize=2,
                    zorder=5,
                )
        axis.axhline(0.0, color="#4B5563", linestyle="--", linewidth=1.0)
        axis.set_xticks(
            np.arange(len(estimators)),
            [_profile_estimator_label(name) for name in estimators],
            rotation=25,
            ha="right",
        )
        axis.set_xlim(-0.65, max(len(estimators) - 0.35, 0.65))
        axis.set_title(metric.display_name)
        axis.grid(True, axis="y", alpha=0.22)

    for empty_index in range(len(metric_specs), row_count * column_count):
        figure.delaxes(axes.flat[empty_index])
    figure.suptitle(
        "Successful vs. Failed Rollout z-scores for Each Method",
        fontsize=15,
    )
    figure.supylabel("Task-normalised geometric error (z-score)")
    figure.supxlabel(
        "Positive values indicate worse-than-average geometry within the same task. "
        "Green/red show successful/failed rollouts; diamonds show mean with "
        "bootstrap 95% CI.",
        fontsize=10,
    )

    paths = []
    for extension in formats:
        path = output_dir / (
            f"estimator_task_normalized_success_vs_failure.{extension}"
        )
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
        paths.append(path)
    plt.close(figure)
    return paths


def build_task_balanced_estimator_rows(
    rollout_rows: Sequence[Dict[str, Any]],
    metric_specs: Sequence[MetricSpec],
) -> list[Dict[str, Any]]:
    """Build an estimator table using equal task weights for every mean."""
    grouped: Dict[Tuple[str, str], list[Dict[str, Any]]] = {}
    for row in rollout_rows:
        key = (
            str(row.get("depth_estimator", "unknown_estimator")),
            str(row.get("task_name", "unknown_task")),
        )
        grouped.setdefault(key, []).append(row)

    estimator_tasks: Dict[str, list[Dict[str, Any]]] = {}
    estimator_rows: Dict[str, list[Dict[str, Any]]] = {}
    for (estimator, task), rows in grouped.items():
        task_row: Dict[str, Any] = {
            "task_name": task,
            "success_rate": float(np.mean([
                float(row["success"]) for row in rows
            ])),
        }
        for metric in metric_specs:
            values = [
                float(row[metric.column])
                for row in rows
                if metric.column in row
            ]
            if values:
                task_row[metric.column] = float(np.mean(values))
        estimator_tasks.setdefault(estimator, []).append(task_row)
        estimator_rows.setdefault(estimator, []).extend(rows)

    output_rows = []
    for estimator, task_rows in sorted(estimator_tasks.items()):
        rows = estimator_rows[estimator]
        success_rate = float(np.mean([
            task_row["success_rate"] for task_row in task_rows
        ]))
        output_row = {
            "depth_estimator": estimator,
            "display_name": _profile_estimator_label(estimator),
            "task_count": len(task_rows),
            "configuration_count": len({
                (
                    row.get("task_name"),
                    row.get("camera_names"),
                    row.get("resolution"),
                    row.get("mask_input"),
                    row.get("eval_name"),
                )
                for row in rows
            }),
            "rollout_count": len(rows),
            "success_count": int(sum(bool(row["success"]) for row in rows)),
            "success_rate": success_rate,
            "success_rate_percent": success_rate * 100.0,
        }
        for metric in metric_specs:
            task_values = [
                float(task_row[metric.column])
                for task_row in task_rows
                if metric.column in task_row
            ]
            output_row[f"{metric.name}_task_count"] = len(task_values)
            output_row[metric.column] = (
                float(np.mean(task_values)) if task_values else None
            )
        output_rows.append(output_row)
    return output_rows


def write_task_balanced_estimator_table(
    path: Path,
    rows: Sequence[Dict[str, Any]],
    metric_specs: Sequence[MetricSpec],
) -> None:
    headers = [
        "depth_estimator",
        "display_name",
        "task_count",
        "configuration_count",
        "rollout_count",
        "success_count",
        "success_rate",
        "success_rate_percent",
    ]
    for metric in metric_specs:
        headers.extend((f"{metric.name}_task_count", metric.column))
    write_csv_atomic(path, rows, headers)


def write_run_table(
    output_dir: Path,
    runs: Sequence[DP3Run],
    metric_specs: Sequence[MetricSpec],
) -> Path:
    rows = []
    for run in sorted(runs, key=lambda item: (item.task_name, item.estimator, item.run_name)):
        row = {
            "task_name": run.task_name,
            "depth_estimator": run.estimator,
            "camera_names": "|".join(run.camera_names),
            "resolution": run.resolution,
            "mask_input": run.mask_input,
            "seed": run.seed,
            "run_name": run.run_name,
            "rollout_count": run.rollout_count,
            "success_count": run.success_count,
            "success_rate": run.success_rate,
            "success_rate_percent": run.success_rate * 100.0,
            "best_stats_json": str(run.stats_path),
        }
        for metric in metric_specs:
            row[metric.column] = _run_metric_mean(run, metric)
        rows.append(row)
    headers = [
        "task_name", "depth_estimator", "camera_names", "resolution",
        "mask_input", "seed", "run_name", "rollout_count", "success_count",
        "success_rate", "success_rate_percent", "best_stats_json",
        *(metric.column for metric in metric_specs),
    ]
    path = output_dir / "dp3_run_summary.csv"
    write_csv_atomic(path, rows, headers)
    return path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Create DP3 metric-vs-success figures from best_stats.json files."
    )
    parser.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=DEFAULT_ANALYSIS_METRICS,
        default=list(DEFAULT_ANALYSIS_METRICS),
        help=(
            "Metrics to analyze (default: Chamfer, accuracy, completeness, "
            "GT-compensated temporal inconsistency when available, and "
            "F-score @ 20 mm)."
        ),
    )
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--estimators", nargs="+", default=None)
    parser.add_argument("--bins", type=int, default=5)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("png", "pdf", "svg"),
        default=["png"],
    )
    parser.add_argument("--no_annotations", action="store_true")
    parser.add_argument("--log_x", action="store_true")
    parser.add_argument(
        "--split-geometric-profile-heatmap",
        "--split_geometric_profile_heatmap",
        "--split-geometric-profile",
        "--split_geometric_profile",
        dest="split_geometric_profile_heatmap",
        action="store_true",
        help=(
            "Write separately normalized geometric-profile heatmaps for "
            "GT/depth estimators and GT/corrupted-depth baselines, plus one vertically "
            "stacked figure containing both independently normalized panels, "
            "instead of the original combined heatmap."
        ),
    )
    parser.add_argument(
        "--bootstrap_samples",
        type=int,
        default=5000,
        help=(
            "Rollout bootstrap samples for success/failure mean confidence "
            "intervals (default: 5000)."
        ),
    )
    parser.add_argument(
        "--bootstrap_seed",
        type=int,
        default=0,
        help="Random seed for rollout bootstrap confidence intervals.",
    )
    parser.add_argument(
        "--cd_fit_type",
        choices=("linear", "curve"),
        default="linear",
        help="Trend-line type for all estimator-level total figures.",
    )
    return parser.parse_args(argv)


def run_analysis(args) -> Dict[str, Any]:
    input_dir = args.input_dir.resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(input_dir)
    if args.bins < 2:
        raise ValueError("--bins must be at least 2")
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive")
    if args.bootstrap_samples <= 0:
        raise ValueError("--bootstrap_samples must be positive")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    estimator_level_dir = output_dir / "estimator_level"
    task_level_dir = output_dir / "task_level"
    for directory in (estimator_level_dir, task_level_dir):
        directory.mkdir(parents=True, exist_ok=True)

    runs, skipped = discover_runs(
        input_dir,
        tasks=set(args.tasks) if args.tasks else None,
        estimators=set(args.estimators) if args.estimators else None,
    )
    if not runs:
        raise RuntimeError(
            f"No processed DP3 best_stats.json files found under {input_dir}"
        )
    metric_specs = build_metric_specs(
        runs,
        requested=args.metrics,
        allow_missing=list(args.metrics) == list(DEFAULT_ANALYSIS_METRICS),
    )
    if not metric_specs:
        raise RuntimeError("No rollout metrics are available for plotting")
    removed_obsolete_outputs = remove_obsolete_metric_outputs(
        output_dir,
        {metric.name for metric in metric_specs},
    )

    rollout_rows = [row for run in runs for row in run.rollout_rows]
    expected_by_config = {
        run.config_key: run.rollout_count for run in runs
    }
    all_paths = []
    metric_payloads = {}
    for metric in metric_specs:
        metric_output_dir = task_level_dir / metric.file_name
        metric_output_dir.mkdir(parents=True, exist_ok=True)
        summaries, grouped_rows = summarize_configurations(
            rollout_rows,
            expected_by_config,
            metric,
        )
        if not summaries:
            raise RuntimeError(f"No rollout values found for {metric.name}")
        global_paths, correlations = plot_global_metric_vs_success(
            summaries,
            metric,
            metric_output_dir,
            args.formats,
            args.dpi,
            annotate=not args.no_annotations,
            log_x=args.log_x,
        )
        task_paths = plot_task_success_curves(
            summaries,
            grouped_rows,
            metric,
            metric_output_dir,
            args.formats,
            args.dpi,
            bin_count=args.bins,
            log_x=args.log_x,
        )
        distribution_paths = plot_success_failure_distributions(
            summaries,
            grouped_rows,
            metric,
            metric_output_dir,
            args.formats,
            args.dpi,
            log_x=args.log_x,
        )
        paths = global_paths + task_paths + distribution_paths
        all_paths.extend(paths)
        metric_payloads[metric.name] = {
            "display_name": metric.display_name,
            "units": metric.units,
            "lower_is_better": metric.lower_is_better,
            "configuration_count": len(summaries),
            "global_correlations": correlations,
            "figures": [str(path) for path in paths],
        }

    success_failure_dir = task_level_dir / "success_vs_failure"
    success_failure_dir.mkdir(parents=True, exist_ok=True)
    normalized_outcomes = task_normalized_outcome_values(
        rollout_rows,
        metric_specs,
    )
    normalized_outcome_rows = summarize_task_normalized_outcomes(
        normalized_outcomes,
        metric_specs,
        args.bootstrap_samples,
        args.bootstrap_seed,
    )
    normalized_outcome_table_path = (
        success_failure_dir / "task_normalized_group_statistics.csv"
    )
    write_task_normalized_outcome_table(
        normalized_outcome_table_path,
        normalized_outcome_rows,
    )
    normalized_outcome_paths = plot_task_normalized_success_failure(
        normalized_outcomes,
        normalized_outcome_rows,
        metric_specs,
        success_failure_dir,
        args.formats,
        args.dpi,
    )
    all_paths.extend(normalized_outcome_paths)

    estimator_success_failure_dir = (
        estimator_level_dir / "success_vs_failure"
    )
    estimator_success_failure_dir.mkdir(parents=True, exist_ok=True)
    estimator_outcome_rows = summarize_estimator_normalized_outcomes(
        normalized_outcomes,
        metric_specs,
        args.bootstrap_samples,
        args.bootstrap_seed,
    )
    estimator_outcome_table_path = (
        estimator_success_failure_dir
        / "estimator_task_normalized_group_statistics.csv"
    )
    write_estimator_normalized_outcome_table(
        estimator_outcome_table_path,
        estimator_outcome_rows,
    )
    estimator_plot_metric_specs = tuple(
        metric
        for metric_name in ESTIMATOR_SUCCESS_FAILURE_METRICS
        for metric in metric_specs
        if metric.name == metric_name
    )
    if estimator_plot_metric_specs:
        estimator_outcome_paths = plot_estimator_normalized_success_failure(
            normalized_outcomes,
            estimator_outcome_rows,
            estimator_plot_metric_specs,
            estimator_success_failure_dir,
            args.formats,
            args.dpi,
        )
    else:
        estimator_outcome_paths = []
    all_paths.extend(estimator_outcome_paths)

    total_metric_figures = {}
    total_metric_status = {}
    total_x_labels = {
        "chamfer": "Average Chamfer Distance (across tasks)",
        "accuracy_mm": "Average Accuracy Distance (mm, across tasks)",
        "completeness_mm": "Average Completeness Distance (mm, across tasks)",
        "temporal_inconsistency_mm": (
            "Average GT-Compensated Temporal Inconsistency "
            "(mm, across tasks)"
        ),
        "fscore_20mm": "Average F-score @ 20 mm (across tasks)",
    }
    for metric in metric_specs:
        stem = TOTAL_METRIC_FILE_NAMES.get(metric.name)
        if stem is None:
            continue
        pairs = total_metric_pairs(runs, metric)
        if len(pairs) < 2:
            total_metric_figures[metric.name] = []
            total_metric_status[metric.name] = (
                f"Skipped {stem} because at least two estimators with "
                f"{metric.display_name} results are required."
            )
            continue
        paths = []
        for extension in args.formats:
            path = estimator_level_dir / f"{stem}.{extension}"
            plot_cd_vs_success_yerr(
                pairs,
                out_png=str(path),
                title={
                    "chamfer": "Average Chamfer Distance vs Success Rate (BC)",
                    "accuracy_mm": (
                        "Average Accuracy distance vs Success Rate (DP3)"
                    ),
                    "temporal_inconsistency_mm": (
                        "Average Temporal Inconsistency vs Success Rate (DP3)"
                    ),
                }.get(
                    metric.name,
                    f"Average {metric.display_name} vs Success Rate (BC)",
                ),
                fit_type=args.cd_fit_type,
                x_label=total_x_labels[metric.name],
                x_divisor=10.0 if metric.name == "chamfer" else 1.0,
                x_as_percent=metric.units == "fraction",
                show_correlations=metric.name != "chamfer",
                annotation_fontsize=(8 if metric.name == "accuracy_mm" else 10),
                annotation_offset_scale=(
                    0.5 if metric.name == "accuracy_mm" else 1.0
                ),
                annotation_offsets=(
                    {
                        "Pi3": (-6, -6),
                        "MoGe": (6, -6),
                        "MapAnything": (-6, -6),
                    }
                    if metric.name == "accuracy_mm"
                    else None
                ),
            )
            paths.append(path)
        all_paths.extend(paths)
        total_metric_figures[metric.name] = [str(path) for path in paths]
        total_metric_status[metric.name] = None

    legacy_paths = total_metric_figures.get("chamfer", [])
    legacy_path = Path(legacy_paths[0]) if legacy_paths else None
    legacy_status = total_metric_status.get(
        "chamfer",
        "Chamfer was not selected for analysis.",
    )

    geometric_profiles = build_estimator_geometric_profiles(rollout_rows)
    geometric_profile_table_path = (
        estimator_level_dir / "geometric_profile_values.csv"
    )
    write_estimator_geometric_profiles(
        geometric_profile_table_path,
        geometric_profiles,
    )
    combined_heatmap_stem = "geometric_profile_heatmap"
    split_heatmap_stems = {
        "geometric_profile_heatmap_depth_estimators",
        "geometric_profile_heatmap_gt_noise",
        "geometric_profile_heatmap_split_combined",
    }
    inactive_heatmap_stems = (
        {combined_heatmap_stem}
        if args.split_geometric_profile_heatmap
        else split_heatmap_stems
    )
    for file_stem in inactive_heatmap_stems:
        for extension in ("png", "pdf", "svg"):
            stale_path = estimator_level_dir / f"{file_stem}.{extension}"
            if stale_path.is_file():
                stale_path.unlink()
                removed_obsolete_outputs.append(stale_path)

    geometric_profile_paths = []
    geometric_profile_status = None
    try:
        if args.split_geometric_profile_heatmap:
            split_specs = (
                (
                    "depth estimators",
                    DEPTH_ESTIMATOR_PROFILE_ORDER,
                    "geometric_profile_heatmap_depth_estimators",
                    "Geometric Profile: Depth Estimators",
                ),
                (
                    "corrupted-depth baselines",
                    GT_NOISE_PROFILE_ORDER,
                    "geometric_profile_heatmap_gt_noise",
                    "Geometric Profile: Corrupted-Depth Baselines",
                ),
            )
            missing_groups = []
            split_group_profiles = {}
            for group_name, estimator_order, file_stem, title in split_specs:
                group_profiles = select_geometric_profile_group(
                    geometric_profiles,
                    estimator_order,
                )
                if not group_profiles:
                    missing_groups.append(group_name)
                    continue
                split_group_profiles[group_name] = group_profiles
                geometric_profile_paths.extend(
                    plot_estimator_geometric_profile_heatmap(
                        group_profiles,
                        estimator_level_dir,
                        args.formats,
                        args.dpi,
                        file_stem=file_stem,
                        title=title,
                        divider_after_estimator=None,
                    )
                )
            if len(split_group_profiles) == len(split_specs):
                geometric_profile_paths.extend(
                    plot_split_geometric_profile_combined_figure(
                        split_group_profiles["depth estimators"],
                        split_group_profiles["corrupted-depth baselines"],
                        estimator_level_dir,
                        args.formats,
                        args.dpi,
                    )
                )
            if missing_groups:
                geometric_profile_status = (
                    "Skipped empty geometric-profile groups: "
                    + ", ".join(missing_groups)
                )
        else:
            geometric_profile_paths = plot_estimator_geometric_profile_heatmap(
                geometric_profiles,
                estimator_level_dir,
                args.formats,
                args.dpi,
            )
    except ValueError as exc:
        geometric_profile_status = str(exc)
    all_paths.extend(geometric_profile_paths)

    estimator_table_path = (
        estimator_level_dir / "dp3_estimator_sr_and_metrics.csv"
    )
    estimator_rows = build_task_balanced_estimator_rows(
        rollout_rows,
        metric_specs,
    )
    write_task_balanced_estimator_table(
        estimator_table_path,
        estimator_rows,
        metric_specs,
    )
    temporal_tradeoff_paths, temporal_tradeoff_status = (
        plot_accuracy_temporal_tradeoff(
            estimator_rows,
            estimator_level_dir,
            args.formats,
            args.dpi,
        )
    )
    all_paths.extend(temporal_tradeoff_paths)
    run_table_path = write_run_table(task_level_dir, runs, metric_specs)
    payload = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "run_count": len(runs),
        "rollout_count": len(rollout_rows),
        "task_count": len({run.task_name for run in runs}),
        "estimator_count": len({run.estimator for run in runs}),
        "metric_count": len(metric_specs),
        "removed_obsolete_metric_outputs": [
            str(path) for path in removed_obsolete_outputs
        ],
        "metrics": metric_payloads,
        "figures": [str(path) for path in all_paths],
        "output_layout": {
            "estimator_level": str(estimator_level_dir),
            "task_level": str(task_level_dir),
        },
        "total_metric_figures": total_metric_figures,
        "total_metric_status": total_metric_status,
        "geometric_profile_heatmap": [
            str(path) for path in geometric_profile_paths
        ],
        "geometric_profile_table": str(geometric_profile_table_path),
        "geometric_profile_status": geometric_profile_status,
        "geometric_profile_mode": (
            "split_independent_normalization"
            if args.split_geometric_profile_heatmap
            else "combined"
        ),
        "geometric_profile_groups": {
            "depth_estimators": list(DEPTH_ESTIMATOR_PROFILE_ORDER),
            "gt_noise_baselines": list(GT_NOISE_PROFILE_ORDER),
        },
        "geometric_profile_aggregation": (
            "For each estimator and metric, rollout values are averaged "
            "within each task and task means are then averaged with equal "
            "weight. Heatmap colors are min-max normalized by column; "
            "accuracy and completeness colors are reversed so higher intensity "
            "always indicates better quality. Cell labels show unnormalized "
            "means. Chamfer remains in the CSV table but is omitted from the "
            "heatmap. In split mode, each estimator group is normalized "
            "independently."
        ),
        "task_normalized_success_failure_figures": [
            str(path) for path in normalized_outcome_paths
        ],
        "task_normalized_success_failure_table": str(
            normalized_outcome_table_path
        ),
        "task_normalized_success_failure_method": (
            "Each rollout metric is z-scored using the mean and population "
            "standard deviation of all selected-estimator rollouts in the "
            "same task before splitting by success/failure. Precision, "
            "recall, and F-score z-scores have their sign reversed so "
            "positive always means worse geometry. Group means use a "
            "rollout-unit percentile bootstrap 95% confidence interval."
        ),
        "estimator_success_failure_figures": [
            str(path) for path in estimator_outcome_paths
        ],
        "estimator_success_failure_table": str(
            estimator_outcome_table_path
        ),
        "estimator_success_failure_method": (
            "Uses the same jointly computed all-estimator task-wise z-scores "
            "as the task-level comparison, then splits rollouts by estimator "
            "and outcome without estimator-specific renormalization. Means "
            "and percentile-bootstrap 95% confidence intervals treat each "
            "rollout as one unit."
        ),
        "legacy_cd_vs_success": (
            str(legacy_path) if legacy_path is not None else None
        ),
        "legacy_cd_vs_success_source": str(
            REPOSITORY_ROOT / "plot_cd_vs_success.py"
        ),
        "legacy_cd_vs_success_status": legacy_status,
        "estimator_table": str(estimator_table_path),
        "estimator_table_aggregation": (
            "Rollout values are averaged within task, then task means are "
            "averaged equally for each estimator."
        ),
        "accuracy_temporal_tradeoff_figures": [
            str(path) for path in temporal_tradeoff_paths
        ],
        "accuracy_temporal_tradeoff_status": temporal_tradeoff_status,
        "run_table": str(run_table_path),
        "metric_units": {
            metric.name: metric.units for metric in metric_specs
        },
        "temporal_inconsistency_definition": (
            "For each consecutive rollout step, predicted and GT point clouds "
            "are transformed from the end-effector frame into the world frame. "
            "The metric is the absolute difference, in millimetres, between "
            "their symmetric mean nearest-neighbor inter-frame motions. "
            "Rollout means are statistical units; lower is better."
        ),
        "skipped_files": skipped,
    }
    summary_path = output_dir / "dp3_analysis_summary.json"
    write_json_atomic(summary_path, payload)
    payload["summary"] = str(summary_path)
    return payload


def main(argv=None):
    summary = run_analysis(parse_args(argv))
    print(
        f"Created {len(summary['figures'])} figure files from "
        f"{summary['rollout_count']} rollouts across "
        f"{summary['run_count']} DP3 runs."
    )
    for path in summary["figures"]:
        print(path)
    if summary["removed_obsolete_metric_outputs"]:
        print(
            "Removed "
            f"{len(summary['removed_obsolete_metric_outputs'])} obsolete "
            "metric figure files."
        )
    if summary["legacy_cd_vs_success_status"]:
        print(summary["legacy_cd_vs_success_status"])
    if summary["geometric_profile_status"]:
        print(summary["geometric_profile_status"])
    if summary["accuracy_temporal_tradeoff_status"]:
        print(summary["accuracy_temporal_tradeoff_status"])
    print(summary["geometric_profile_table"])
    print(summary["task_normalized_success_failure_table"])
    print(summary["estimator_success_failure_table"])
    print(summary["estimator_table"])
    print(summary["run_table"])
    print(summary["summary"])


if __name__ == "__main__":
    main()
