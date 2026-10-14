#!/usr/bin/env python3
"""Process Instant Policy RLBench rollout artifacts and maintain an XLSX report.

The processor is intentionally separate from ``process_dp3_rollout_artifacts``.
It recursively scans each IP configuration's canonical ``eval_0000`` run,
processes every atomically paired rollout snapshot (including incomplete runs),
writes per-rollout metric JSON/video
diagnostics, and rebuilds one workbook containing:

* the full expected experiment grid, including blank unevaluated rows;
* every discovered configuration evaluation;
* every processed rollout and its metric summaries; and
* metric definitions and normalization settings.

The XLSX writer uses only the Python standard library so the script does not
require openpyxl or xlsxwriter.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import tempfile
import warnings
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Tuple
from xml.sax.saxutils import escape

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from tqdm import tqdm

if __package__ in (None, ""):
    from chamfer_metrics import (
        chamfer_kdtree,
        directional_nearest_neighbor_distances,
        extract_xyz,
        normalize_pair,
        summarize,
    )
    from rollout_artifacts import validate_eval_artifacts
else:
    from .chamfer_metrics import (
        chamfer_kdtree,
        directional_nearest_neighbor_distances,
        extract_xyz,
        normalize_pair,
        summarize,
    )
    from .rollout_artifacts import validate_eval_artifacts


DEFAULT_TASKS = (
    "phone_on_base",
    "open_box",
    "close_box",
    "plate_out",
    "basketball",
    "close_microwave",
    "toilet_seat_down",
    "toilet_seat_up",
    "toilet_roll_off",
    "meat_off_grill",
)
DEFAULT_ESTIMATORS = ("gt", "vggt", "map_anything", "mast3r", "moge", "pi3")
DEFAULT_CAMERA_SETUPS = (
    ("front",),
    ("front", "left_shoulder"),
    ("front", "left_shoulder", "wrist"),
    ("front", "left_shoulder", "right_shoulder", "wrist", "overhead"),
)
DEFAULT_RESOLUTIONS = (128, 256, 512)
SUMMARY_FIELDS = ("count", "mean", "median", "std", "p95")
XLSX_MAX_ROWS = 1_048_576
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = REPOSITORY_ROOT / "results" / "ip_eval"


@dataclass(frozen=True)
class MetricDefinition:
    name: str
    display_name: str
    required_keys: Sequence[str]
    compute: Callable[
        [Dict[str, np.ndarray], argparse.Namespace, Dict[str, Any]],
        np.ndarray,
    ]
    description: str
    units: str


@dataclass(frozen=True)
class ConfigKey:
    task_name: str
    depth_estimator: str
    camera_names: Tuple[str, ...]
    resolution: int
    mask_input: Optional[bool]

    @property
    def camera_text(self) -> str:
        return "|".join(self.camera_names)

    @property
    def mask_text(self) -> str:
        if self.mask_input is None:
            return "none"
        return "true" if self.mask_input else "false"

    def sort_key(self) -> Tuple[Any, ...]:
        return (
            self.task_name,
            self.depth_estimator,
            self.camera_names,
            self.resolution,
            self.mask_text,
        )


@dataclass
class RolloutRecord:
    npz_path: Path
    outcome: bool
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    video_error: Optional[str] = None


@dataclass
class EvalRun:
    config: ConfigKey
    eval_dir: Path
    manifest: Dict[str, Any]
    manifest_status: str
    rollout_paths: Tuple[Path, ...]
    outcomes: Tuple[bool, ...]
    artifact_pair_count: int
    expected_rollouts: Optional[int]
    validation_error: Optional[str] = None
    rollout_records: list[RolloutRecord] = field(default_factory=list)
    stats: Optional[Dict[str, Any]] = None
    processing_error: Optional[str] = None

    @property
    def is_valid_complete(self) -> bool:
        return (
            self.expected_rollouts is not None
            and self.rollout_count == self.expected_rollouts
            and self.validation_error is None
        )

    @property
    def is_processable(self) -> bool:
        """Whether the scan captured at least one complete NPZ/MP4 pair."""
        return self.rollout_count > 0

    @property
    def success_count(self) -> int:
        return int(sum(self.outcomes))

    @property
    def failure_count(self) -> int:
        return len(self.outcomes) - self.success_count

    @property
    def rollout_count(self) -> int:
        return len(self.outcomes)

    @property
    def success_rate(self) -> Optional[float]:
        if self.rollout_count == 0:
            return None
        return self.success_count / self.rollout_count


METRIC_REGISTRY: Dict[str, MetricDefinition] = {}


def register_metric(
    name,
    display_name,
    required_keys,
    description,
    units="unitless",
):
    """Decorator used to add metrics without changing the artifact pipeline."""
    def decorator(fn):
        if name in METRIC_REGISTRY:
            raise ValueError(f"Metric {name!r} is already registered")
        METRIC_REGISTRY[name] = MetricDefinition(
            name=name,
            display_name=display_name,
            required_keys=tuple(required_keys),
            compute=fn,
            description=description,
            units=units,
        )
        return fn
    return decorator


@register_metric(
    name="chamfer",
    display_name="Chamfer",
    required_keys=("obs_pointcloud", "gt_pointcloud"),
    description=(
        "Symmetric unsquared nearest-neighbor L2 Chamfer distance after "
        "same-step point-cloud normalization."
    ),
    units="normalized distance",
)
def compute_chamfer(data, args, cache):
    observed = extract_xyz(data["obs_pointcloud"])
    reference = extract_xyz(data["gt_pointcloud"])
    if observed.shape[0] != reference.shape[0]:
        raise ValueError(
            "Chamfer frame-count mismatch: "
            f"observed={observed.shape[0]}, reference={reference.shape[0]}"
        )

    values = []
    for frame_idx in tqdm(
        range(observed.shape[0]),
        desc="Computing Chamfer",
        leave=False,
        disable=True,
    ):
        pred_normalized, gt_normalized = normalize_pair(
            observed[frame_idx],
            reference[frame_idx],
            pair_mode=args.pair_mode,
            norm_mode=args.norm_mode,
            robust=args.robust_center,
        )
        values.append(chamfer_kdtree(pred_normalized, gt_normalized))
    return np.asarray(values, dtype=np.float64)


def _raw_directional_distances(data, cache):
    """Return cached predicted→GT and GT→predicted distances in meters."""
    cache_key = "raw_directional_distances_m"
    if cache_key in cache:
        return cache[cache_key]

    observed = extract_xyz(data["obs_pointcloud"])
    reference = extract_xyz(data["gt_pointcloud"])
    if observed.shape[0] != reference.shape[0]:
        raise ValueError(
            "Point-cloud frame-count mismatch: "
            f"observed={observed.shape[0]}, reference={reference.shape[0]}"
        )

    predicted_to_gt = []
    gt_to_predicted = []
    for frame_index in range(observed.shape[0]):
        accuracy_distances, completeness_distances = (
            directional_nearest_neighbor_distances(
                observed[frame_index],
                reference[frame_index],
            )
        )
        predicted_to_gt.append(accuracy_distances)
        gt_to_predicted.append(completeness_distances)
    cache[cache_key] = (
        np.stack(predicted_to_gt, axis=0),
        np.stack(gt_to_predicted, axis=0),
    )
    return cache[cache_key]


@register_metric(
    name="accuracy_mm",
    display_name="Accuracy / flying-points distance",
    required_keys=("obs_pointcloud", "gt_pointcloud"),
    description=(
        "Mean predicted-to-GT nearest-neighbor L2 distance in millimeters. "
        "Larger values indicate inaccurate or flying predicted points."
    ),
    units="mm",
)
def compute_accuracy_mm(data, args, cache):
    predicted_to_gt, _ = _raw_directional_distances(data, cache)
    return predicted_to_gt.mean(axis=1) * 1000.0


@register_metric(
    name="completeness_mm",
    display_name="Completeness / missing-geometry distance",
    required_keys=("obs_pointcloud", "gt_pointcloud"),
    description=(
        "Mean GT-to-predicted nearest-neighbor L2 distance in millimeters. "
        "Larger values indicate GT geometry missing from the prediction."
    ),
    units="mm",
)
def compute_completeness_mm(data, args, cache):
    _, gt_to_predicted = _raw_directional_distances(data, cache)
    return gt_to_predicted.mean(axis=1) * 1000.0


def _threshold_metric(metric_kind, threshold_mm):
    threshold_m = float(threshold_mm) / 1000.0

    def compute(data, args, cache):
        cache_key = ("threshold_rates", int(threshold_mm))
        if cache_key not in cache:
            predicted_to_gt, gt_to_predicted = _raw_directional_distances(
                data,
                cache,
            )
            precision = (predicted_to_gt <= threshold_m).mean(axis=1)
            recall = (gt_to_predicted <= threshold_m).mean(axis=1)
            denominator = precision + recall
            fscore = np.divide(
                2.0 * precision * recall,
                denominator,
                out=np.zeros_like(denominator, dtype=np.float64),
                where=denominator > 0.0,
            )
            cache[cache_key] = {
                "precision": precision,
                "recall": recall,
                "fscore": fscore,
            }
        return cache[cache_key][metric_kind]

    return compute


for _threshold_mm in (5, 10, 20):
    register_metric(
        name=f"precision_{_threshold_mm}mm",
        display_name=f"Precision @ {_threshold_mm} mm",
        required_keys=("obs_pointcloud", "gt_pointcloud"),
        description=(
            "Fraction of predicted points whose nearest GT point is within "
            f"{_threshold_mm} mm; one minus this value is the flying-point rate."
        ),
        units="fraction",
    )(_threshold_metric("precision", _threshold_mm))
    register_metric(
        name=f"recall_{_threshold_mm}mm",
        display_name=f"Recall @ {_threshold_mm} mm",
        required_keys=("obs_pointcloud", "gt_pointcloud"),
        description=(
            "Fraction of GT points whose nearest predicted point is within "
            f"{_threshold_mm} mm; one minus this value is the missing-geometry rate."
        ),
        units="fraction",
    )(_threshold_metric("recall", _threshold_mm))
    register_metric(
        name=f"fscore_{_threshold_mm}mm",
        display_name=f"F-score @ {_threshold_mm} mm",
        required_keys=("obs_pointcloud", "gt_pointcloud"),
        description=(
            f"Harmonic mean of precision and recall at {_threshold_mm} mm."
        ),
        units="fraction",
    )(_threshold_metric("fscore", _threshold_mm))


DEFAULT_METRICS = tuple(METRIC_REGISTRY)


def _scalar(data: Dict[str, np.ndarray], key: str, default=None):
    if key not in data:
        return default
    value = np.asarray(data[key])
    if value.size != 1:
        raise ValueError(f"{key} must be scalar, got shape {value.shape}")
    return value.item()


def load_rollout_data(npz_path):
    """Load and strictly validate one IP rollout state archive."""
    with np.load(npz_path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}

    required = ("agent_pos", "T_w_e", "obs_pointcloud", "gt_pointcloud")
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"{npz_path} is missing required keys: {missing}")

    agent_pos = np.asarray(data["agent_pos"])
    transforms = np.asarray(data["T_w_e"])
    observed = np.asarray(data["obs_pointcloud"])
    reference = np.asarray(data["gt_pointcloud"])
    if agent_pos.ndim != 2 or agent_pos.shape[1] != 7:
        raise ValueError(f"agent_pos must have shape (T, 7), got {agent_pos.shape}")
    frame_count = agent_pos.shape[0]
    if transforms.shape != (frame_count, 4, 4):
        raise ValueError(
            f"T_w_e must have shape ({frame_count}, 4, 4), got {transforms.shape}"
        )
    expected_cloud_shape = (frame_count, 2048, 3)
    if observed.shape != expected_cloud_shape:
        raise ValueError(
            f"obs_pointcloud must have shape {expected_cloud_shape}, "
            f"got {observed.shape}"
        )
    if reference.shape != expected_cloud_shape:
        raise ValueError(
            f"gt_pointcloud must have shape {expected_cloud_shape}, "
            f"got {reference.shape}"
        )

    for name, array in (
        ("agent_pos", agent_pos),
        ("T_w_e", transforms),
        ("obs_pointcloud", observed),
        ("gt_pointcloud", reference),
    ):
        if array.dtype != np.float32:
            raise ValueError(f"{name} must have dtype float32, got {array.dtype}")
        if not np.isfinite(array).all():
            raise ValueError(f"{npz_path}: {name} contains NaN or Inf values")

    if "decision_video_frame_indices" in data:
        indices = np.asarray(data["decision_video_frame_indices"])
        if indices.shape != (frame_count,):
            raise ValueError(
                "decision_video_frame_indices must have shape "
                f"({frame_count},), got {indices.shape}"
            )
        if np.any(indices < 0) or np.any(np.diff(indices) <= 0):
            raise ValueError(
                "decision_video_frame_indices must be nonnegative and "
                "strictly increasing"
            )
    return data


def validate_rollout_identity(data, npz_path: Path, config: ConfigKey) -> None:
    """Ensure archive metadata cannot be attributed to the wrong table row."""
    expected_scalars = {
        "task_name": config.task_name,
        "depth_estimator": config.depth_estimator,
        "resolution": config.resolution,
    }
    for key, expected in expected_scalars.items():
        if key not in data:
            raise KeyError(f"{npz_path} is missing identity key {key!r}")
        actual = _scalar(data, key)
        if str(actual) != str(expected):
            raise ValueError(
                f"{npz_path}: {key}={actual!r} does not match {expected!r}"
            )

    cameras = tuple(str(name) for name in np.asarray(data.get("camera_names", ())))
    if cameras != config.camera_names:
        raise ValueError(
            f"{npz_path}: camera_names={cameras!r} does not match "
            f"{config.camera_names!r}"
        )
    if "mask_input" not in data:
        raise KeyError(f"{npz_path} is missing identity key 'mask_input'")
    archive_mask = normalize_mask_input(_scalar(data, "mask_input"))
    if archive_mask != config.mask_input:
        raise ValueError(
            f"{npz_path}: mask_input={archive_mask!r} does not match "
            f"{config.mask_input!r}"
        )

    match = re.fullmatch(r"rollout_(\d+)_(success|failure)", npz_path.stem)
    if match is None:
        raise ValueError(f"Invalid rollout filename: {npz_path.name}")
    archive_rollout_id = int(_scalar(data, "rollout_id"))
    filename_rollout_id = int(match.group(1))
    if archive_rollout_id != filename_rollout_id:
        raise ValueError(
            f"{npz_path}: rollout_id={archive_rollout_id} does not match "
            f"filename ID {filename_rollout_id}"
        )


def compute_metrics(data, metric_names, args):
    frame_count = int(data["agent_pos"].shape[0])
    results = {}
    cache = {}
    for name in metric_names:
        definition = METRIC_REGISTRY[name]
        missing = [key for key in definition.required_keys if key not in data]
        if missing:
            raise KeyError(
                f"Metric {name!r} requires keys {missing}; available keys: "
                f"{sorted(data)}"
            )
        values = np.asarray(
            definition.compute(data, args, cache),
            dtype=np.float64,
        )
        if values.shape != (frame_count,):
            raise ValueError(
                f"Metric {name!r} returned shape {values.shape}; "
                f"expected ({frame_count},)"
            )
        if not np.isfinite(values).all():
            raise ValueError(f"Metric {name!r} returned non-finite values")
        results[name] = {
            "display_name": definition.display_name,
            "description": definition.description,
            "units": definition.units,
            "values": values,
            "summary": summarize(values),
        }
    return results


def rollout_outcome(npz_path):
    outcome = npz_path.stem.rsplit("_", maxsplit=1)[-1].lower()
    if outcome == "success":
        return True
    if outcome == "failure":
        return False
    raise ValueError(
        f"Cannot determine rollout outcome from {npz_path.name!r}; expected "
        "the filename to end in '_success.npz' or '_failure.npz'."
    )


def normalize_mask_input(value) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text in ("none", "na", "n/a", "null", ""):
        return None
    if text in ("true", "1", "yes", "masked"):
        return True
    if text in ("false", "0", "no", "unmasked"):
        return False
    raise ValueError(f"Unsupported mask_input value: {value!r}")


def parse_camera_setup(value: str) -> Tuple[str, ...]:
    names = tuple(part for part in re.split(r"[+,|]", value) if part)
    if not names:
        raise argparse.ArgumentTypeError("Camera setup must contain a camera name")
    return names


def config_from_path(eval_dir: Path) -> ConfigKey:
    config_dir = eval_dir.parent.parent
    mask_component = config_dir.name.removeprefix("mask-")
    resolution_component = config_dir.parent.name.removeprefix("res-")
    camera_component = config_dir.parent.parent.name.removeprefix("cams-")
    estimator = config_dir.parent.parent.parent.name
    task_name = config_dir.parent.parent.parent.parent.name
    return ConfigKey(
        task_name=task_name,
        depth_estimator=estimator,
        camera_names=tuple(camera_component.split("+")),
        resolution=int(resolution_component),
        mask_input=normalize_mask_input(mask_component),
    )


def config_from_manifest(manifest: Dict[str, Any], fallback: ConfigKey) -> ConfigKey:
    cameras = manifest.get("camera_names", fallback.camera_names)
    if isinstance(cameras, str):
        cameras = parse_camera_setup(cameras)
    else:
        cameras = tuple(str(name) for name in cameras)
    return ConfigKey(
        task_name=str(manifest.get("task_name", fallback.task_name)),
        depth_estimator=str(
            manifest.get("depth_estimator", fallback.depth_estimator)
        ),
        camera_names=tuple(cameras),
        resolution=int(manifest.get("resolution", fallback.resolution)),
        mask_input=normalize_mask_input(
            manifest.get("mask_input", fallback.mask_input)
        ),
    )


def discover_eval_runs(
    input_root: Path,
    pattern="rollout_*.npz",
    expected_rollouts=200,
) -> list[EvalRun]:
    """Snapshot each configuration's single canonical evaluation directory."""
    input_root = Path(input_root)
    eval_dirs = sorted(
        path for path in input_root.rglob("eval_*")
        if (
            path.is_dir()
            and path.parent.name == "rollouts"
            and path.name == "eval_0000"
        )
    )
    runs = []
    for eval_dir in eval_dirs:
        try:
            fallback_config = config_from_path(eval_dir)
        except (ValueError, IndexError) as exc:
            warnings.warn(f"Skipping unrecognized eval path {eval_dir}: {exc}")
            continue

        manifest_path = eval_dir / "manifest.json"
        manifest = {}
        validation_issues = []
        if not manifest_path.is_file():
            manifest_status = "missing_manifest"
        else:
            try:
                with manifest_path.open("r", encoding="utf-8") as file:
                    manifest = json.load(file)
                if not isinstance(manifest, dict):
                    raise TypeError("manifest root is not an object")
                manifest_status = str(manifest.get("status", "unknown"))
            except (OSError, ValueError, TypeError) as exc:
                manifest = {}
                manifest_status = "invalid_manifest"
                warnings.warn(f"Ignoring unreadable manifest {manifest_path}: {exc}")

        config = fallback_config
        run_expected_rollouts = expected_rollouts
        recorded_rollouts = manifest.get("recorded_rollouts")
        if isinstance(recorded_rollouts, int) and recorded_rollouts > 0:
            run_expected_rollouts = recorded_rollouts
        else:
            iterations = manifest.get("num_demo_iterations")
            rollouts_per_demo = manifest.get("num_rollouts_per_demo")
            if (
                isinstance(iterations, int)
                and iterations > 0
                and isinstance(rollouts_per_demo, int)
                and rollouts_per_demo > 0
            ):
                run_expected_rollouts = iterations * rollouts_per_demo

        state_dir = eval_dir / "live_states"
        video_dir = eval_dir / "live_videos"
        state_paths = tuple(sorted(state_dir.glob(pattern))) \
            if state_dir.is_dir() else ()
        video_stems = {
            path.stem for path in video_dir.glob("rollout_*.mp4")
        } if video_dir.is_dir() else set()
        paired_paths = tuple(
            path for path in state_paths if path.stem in video_stems
        )
        artifact_pair_count = len(paired_paths)

        rollout_paths = []
        outcomes = []
        for path in paired_paths:
            try:
                outcome = rollout_outcome(path)
            except ValueError as exc:
                validation_issues.append(str(exc))
                continue
            rollout_paths.append(path)
            outcomes.append(outcome)

        valid, reason = validate_eval_artifacts(
            eval_dir,
            expected_rollouts=run_expected_rollouts,
        )
        if not valid:
            validation_issues.append(reason)
        elif len(rollout_paths) != run_expected_rollouts:
            validation_issues.append(
                f"pattern {pattern!r} selected {len(rollout_paths)} of "
                f"{run_expected_rollouts} rollouts"
            )

        runs.append(EvalRun(
            config=config,
            eval_dir=eval_dir,
            manifest=manifest,
            manifest_status=manifest_status,
            rollout_paths=tuple(rollout_paths),
            outcomes=tuple(outcomes),
            artifact_pair_count=artifact_pair_count,
            expected_rollouts=run_expected_rollouts,
            validation_error="; ".join(validation_issues)
                if validation_issues else None,
        ))
    return runs


def pose_transform(agent_pos):
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rot.from_rotvec(agent_pos[3:6]).as_matrix()
    transform[:3, 3] = agent_pos[:3]
    return transform


def point_clouds_in_world(data):
    point_clouds = extract_xyz(data["obs_pointcloud"]).astype(np.float64)
    frame_name = str(np.asarray(data.get("point_cloud_frame", "eef")).item())
    if frame_name not in ("eef", "world"):
        raise ValueError(f"Unsupported point_cloud_frame: {frame_name!r}")

    if "T_w_e" in data:
        transforms_array = np.asarray(data["T_w_e"], dtype=np.float64)
        if transforms_array.shape != (point_clouds.shape[0], 4, 4):
            raise ValueError(
                "T_w_e does not match point-cloud frame count: "
                f"{transforms_array.shape}"
            )
        transforms = [transform.copy() for transform in transforms_array]
    else:
        transforms = [pose_transform(state) for state in data["agent_pos"]]

    if frame_name == "world":
        return point_clouds, transforms
    world_clouds = np.empty_like(point_clouds)
    for frame_idx, (points, transform) in enumerate(zip(point_clouds, transforms)):
        world_clouds[frame_idx] = (
            points @ transform[:3, :3].T + transform[:3, 3]
        )
    return world_clouds, transforms


def equal_world_bounds(world_clouds, transforms):
    positions = np.asarray([transform[:3, 3] for transform in transforms])
    minimum = np.minimum(world_clouds.min(axis=(0, 1)), positions.min(axis=0))
    maximum = np.maximum(world_clouds.max(axis=(0, 1)), positions.max(axis=0))
    center = (minimum + maximum) / 2.0
    span = max(float(np.max(maximum - minimum)), 1e-3)
    half_span = span * 0.58
    return [
        (center[axis] - half_span, center[axis] + half_span)
        for axis in range(3)
    ]


def metric_overlay_text(metric_results, frame_idx):
    lines = []
    for result in metric_results.values():
        value = float(result["values"][frame_idx])
        if result["units"] == "fraction":
            formatted_value = f"{value * 100.0:.1f}%"
        elif result["units"] == "mm":
            formatted_value = f"{value:.2f} mm"
        else:
            formatted_value = f"{value:.6f}"
        lines.append(f"{result['display_name']}: {formatted_value}")
    return "\n".join(lines)


def create_video_writer(path, fps, frame_size):
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        tuple(int(value) for value in frame_size),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {path}")
    return writer


def render_point_cloud_video(data, metric_results, output_path, args, fps):
    world_clouds, transforms = point_clouds_in_world(data)
    bounds = equal_world_bounds(world_clouds, transforms)
    frame_count = world_clouds.shape[0]
    dpi = 100
    figure = plt.figure(
        figsize=(args.render_width / dpi, args.render_height / dpi),
        dpi=dpi,
        facecolor="white",
    )
    axis = figure.add_subplot(111, projection="3d")
    figure.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    writer = create_video_writer(
        output_path,
        fps=fps,
        frame_size=(args.render_width, args.render_height),
    )

    try:
        for frame_idx in tqdm(
            range(frame_count),
            desc="Rendering point clouds",
            leave=False,
            disable=getattr(args, "quiet", False),
        ):
            axis.clear()
            points = world_clouds[frame_idx]
            transform = transforms[frame_idx]
            state = data["agent_pos"][frame_idx]
            axis.scatter(
                points[:, 0], points[:, 1], points[:, 2],
                s=args.point_size,
                c="#4C78A8",
                alpha=0.9,
                depthshade=False,
            )
            origin = transform[:3, 3]
            rotation = transform[:3, :3]
            for axis_idx, color in enumerate(("#D62728", "#2CA02C", "#1F77B4")):
                endpoint = origin + rotation[:, axis_idx] * args.agent_axis_size
                axis.plot(
                    [origin[0], endpoint[0]],
                    [origin[1], endpoint[1]],
                    [origin[2], endpoint[2]],
                    color=color,
                    linewidth=3.0,
                )
            axis.set_xlim(*bounds[0])
            axis.set_ylim(*bounds[1])
            axis.set_zlim(*bounds[2])
            axis.set_box_aspect((1.0, 1.0, 1.0))
            axis.view_init(elev=args.elevation, azim=args.azimuth)
            axis.set_xlabel("World X")
            axis.set_ylabel("World Y")
            axis.set_zlabel("World Z")
            axis.set_facecolor("white")
            for pane in (axis.xaxis.pane, axis.yaxis.pane, axis.zaxis.pane):
                pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
            axis.grid(True, color="#DDDDDD", linewidth=0.6)
            pose_text = np.array2string(
                state,
                precision=3,
                suppress_small=True,
                separator=", ",
            )
            overlay = (
                f"Step: {frame_idx}/{frame_count - 1}\n"
                f"agent_pos: {pose_text}"
            )
            metric_text = metric_overlay_text(metric_results, frame_idx)
            if metric_text:
                overlay = f"{overlay}\n{metric_text}"
            axis.text2D(
                0.02,
                0.98,
                overlay,
                transform=axis.transAxes,
                va="top",
                ha="left",
                fontsize=9,
                color="black",
                bbox={
                    "boxstyle": "round,pad=0.4",
                    "facecolor": "white",
                    "edgecolor": "#999999",
                    "alpha": 0.92,
                },
            )
            figure.canvas.draw()
            rgba = np.asarray(figure.canvas.buffer_rgba())
            rgb = np.ascontiguousarray(rgba[:, :, :3])
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
        plt.close(figure)


def resize_to_height(frame, target_height):
    height, width = frame.shape[:2]
    target_width = max(1, int(round(width * target_height / height)))
    return cv2.resize(
        frame,
        (target_width, target_height),
        interpolation=cv2.INTER_AREA,
    )


def stitch_videos(rgb_path, diagnostic_path, output_path, rgb_frame_indices=None):
    """Stitch decision-aligned RGB frames beside the diagnostic video."""
    rgb_capture = cv2.VideoCapture(str(rgb_path))
    diagnostic_capture = cv2.VideoCapture(str(diagnostic_path))
    if not rgb_capture.isOpened():
        raise RuntimeError(f"Could not open rollout video: {rgb_path}")
    if not diagnostic_capture.isOpened():
        rgb_capture.release()
        raise RuntimeError(f"Could not open diagnostic video: {diagnostic_path}")

    rgb_frames = int(round(rgb_capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    diagnostic_frames = int(round(
        diagnostic_capture.get(cv2.CAP_PROP_FRAME_COUNT)
    ))
    fps = rgb_capture.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        fps = diagnostic_capture.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        fps = 10.0

    if rgb_frame_indices is None:
        indices = np.arange(min(rgb_frames, diagnostic_frames), dtype=np.int64)
    else:
        indices = np.asarray(rgb_frame_indices, dtype=np.int64)
        if indices.shape != (diagnostic_frames,):
            raise ValueError(
                "Decision/video mapping does not match diagnostic frames: "
                f"mapping={indices.shape}, diagnostic={diagnostic_frames}"
            )
        if np.any(indices < 0) or np.any(indices >= rgb_frames):
            raise ValueError(
                "decision_video_frame_indices contains an out-of-range frame"
            )
        if np.any(np.diff(indices) <= 0):
            raise ValueError(
                "decision_video_frame_indices must be strictly increasing"
            )

    writer = None
    written_frames = 0
    try:
        for rgb_frame_index in indices:
            rgb_capture.set(cv2.CAP_PROP_POS_FRAMES, int(rgb_frame_index))
            rgb_ok, rgb_frame = rgb_capture.read()
            diagnostic_ok, diagnostic_frame = diagnostic_capture.read()
            if not rgb_ok or not diagnostic_ok:
                raise RuntimeError(
                    "Failed to decode synchronized RGB/diagnostic video frames"
                )
            target_height = max(rgb_frame.shape[0], diagnostic_frame.shape[0])
            rgb_frame = resize_to_height(rgb_frame, target_height)
            diagnostic_frame = resize_to_height(diagnostic_frame, target_height)
            stitched = np.concatenate((rgb_frame, diagnostic_frame), axis=1)
            if writer is None:
                writer = create_video_writer(
                    output_path,
                    fps=fps,
                    frame_size=(stitched.shape[1], stitched.shape[0]),
                )
            writer.write(stitched)
            written_frames += 1
    finally:
        rgb_capture.release()
        diagnostic_capture.release()
        if writer is not None:
            writer.release()

    if written_frames == 0:
        raise RuntimeError(f"No synchronized frames found for {rgb_path}")
    if rgb_frame_indices is None and rgb_frames != diagnostic_frames:
        warnings.warn(
            "Video frame-count mismatch; stitched the shared prefix only: "
            f"rgb={rgb_frames}, diagnostic={diagnostic_frames}",
            stacklevel=2,
        )
    return written_frames


def write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.stem}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as file:
            temp_path = Path(file.name)
            json.dump(payload, file, indent=2, sort_keys=True)
            file.write("\n")
        os.replace(temp_path, path)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def json_metric_results(npz_path, data, metric_results, args, outcome):
    metrics = {}
    for name, result in metric_results.items():
        metrics[name] = {
            "display_name": result["display_name"],
            "description": result["description"],
            "units": result["units"],
            "summary": result["summary"],
            "per_frame": result["values"].tolist(),
        }
        if name == "chamfer":
            metrics[name]["normalization"] = {
                "pair_mode": args.pair_mode,
                "norm_mode": args.norm_mode,
                "robust_center": args.robust_center,
            }
    stat = npz_path.stat()
    return {
        "source_npz": str(npz_path),
        "source_size": int(stat.st_size),
        "source_mtime_ns": int(stat.st_mtime_ns),
        "success": bool(outcome),
        "outcome": "success" if outcome else "failure",
        "frame_count": int(data["agent_pos"].shape[0]),
        "rollout_id": int(_scalar(data, "rollout_id", -1)),
        "demo_iter": int(_scalar(data, "demo_iter", -1)),
        "rollout_in_demo": int(_scalar(data, "rollout_in_demo", -1)),
        "metrics": metrics,
    }


def process_rollout(npz_path, run: EvalRun, args, outcome):
    data = load_rollout_data(npz_path)
    validate_rollout_identity(data, npz_path, run.config)
    if "success" not in data:
        raise KeyError(f"{npz_path} is missing identity key 'success'")
    stored_success = _scalar(data, "success")
    if bool(stored_success) != bool(outcome):
        raise ValueError(
            f"Success label mismatch for {npz_path}: filename={outcome}, "
            f"archive={bool(stored_success)}"
        )
    metric_results = compute_metrics(data, args.metrics, args)
    fps = float(np.asarray(data.get("fps", 10.0)).item())
    stem = npz_path.stem
    metric_path = run.eval_dir / "metrics" / f"{stem}.json"
    diagnostic_path = (
        run.eval_dir / "pointcloud_videos" / f"{stem}_pointcloud.mp4"
    )
    stitched_path = run.eval_dir / "stitched_videos" / f"{stem}_stitched.mp4"
    result = json_metric_results(
        npz_path,
        data,
        metric_results,
        args,
        outcome=outcome,
    )
    write_json_atomic(metric_path, result)

    video_error = None
    if not args.skip_videos:
        try:
            render_point_cloud_video(
                data,
                metric_results,
                output_path=diagnostic_path,
                args=args,
                fps=fps,
            )
            rgb_path = run.eval_dir / "live_videos" / f"{stem}.mp4"
            if not rgb_path.is_file():
                raise FileNotFoundError(
                    f"No matching RGB rollout video: {rgb_path}"
                )
            stitch_videos(
                rgb_path,
                diagnostic_path,
                stitched_path,
                rgb_frame_indices=data.get("decision_video_frame_indices"),
            )
        except Exception as exc:
            video_error = f"{type(exc).__name__}: {exc}"
            result["video_error"] = video_error
            write_json_atomic(metric_path, result)
    return result, video_error


def summarize_metric_group(processed_rollouts, metric_name):
    frame_values = np.concatenate([
        np.asarray(
            rollout["metrics"][metric_name]["per_frame"],
            dtype=np.float64,
        )
        for rollout in processed_rollouts
    ])
    rollout_means = np.asarray([
        rollout["metrics"][metric_name]["summary"]["mean"]
        for rollout in processed_rollouts
    ], dtype=np.float64)
    return {
        "all_frames": summarize(frame_values),
        "rollout_means": summarize(rollout_means),
    }


def build_eval_stats(run: EvalRun, processed_rollouts, args):
    metrics = {}
    for metric_name in args.metrics:
        definition = METRIC_REGISTRY[metric_name]
        metric_stats = {
            "display_name": definition.display_name,
            "description": definition.description,
            "units": definition.units,
            **summarize_metric_group(processed_rollouts, metric_name),
        }
        for success, group_name in (
            (True, "successful_rollouts"),
            (False, "failed_rollouts"),
        ):
            group = [
                rollout for rollout in processed_rollouts
                if rollout["success"] is success
            ]
            if group:
                metric_stats[group_name] = summarize_metric_group(
                    group,
                    metric_name,
                )
        if metric_name == "chamfer":
            metric_stats["normalization"] = {
                "pair_mode": args.pair_mode,
                "norm_mode": args.norm_mode,
                "robust_center": args.robust_center,
            }
        metrics[metric_name] = metric_stats

    return {
        "task_name": run.config.task_name,
        "depth_estimator": run.config.depth_estimator,
        "camera_names": list(run.config.camera_names),
        "resolution": run.config.resolution,
        "mask_input": run.config.mask_input,
        "eval_dir": str(run.eval_dir),
        "eval_name": run.eval_dir.name,
        "success_count": run.success_count,
        "failure_count": run.failure_count,
        "rollout_count": run.rollout_count,
        "success_rate": run.success_rate,
        "success_rate_percent": run.success_rate * 100.0,
        "total_policy_decisions": int(sum(
            rollout["frame_count"] for rollout in processed_rollouts
        )),
        "metrics": metrics,
        "rollouts": processed_rollouts,
    }


def process_all_eval_runs(runs: Sequence[EvalRun], args) -> list[str]:
    errors = []
    processable_runs = [run for run in runs if run.is_processable]
    for run in tqdm(
        processable_runs,
        desc="IP evaluations",
        disable=args.quiet,
    ):
        run.rollout_records = [
            RolloutRecord(path, outcome)
            for path, outcome in zip(run.rollout_paths, run.outcomes)
        ]
        processed = []
        for record in tqdm(
            run.rollout_records,
            desc=run.eval_dir.name,
            leave=False,
            disable=args.quiet,
        ):
            try:
                record.result, record.video_error = process_rollout(
                    record.npz_path,
                    run,
                    args,
                    record.outcome,
                )
                processed.append(record.result)
                if record.video_error:
                    errors.append(
                        f"{record.npz_path} video: {record.video_error}"
                    )
            except Exception as exc:  # Continue so every rollout is attempted.
                record.error = f"{type(exc).__name__}: {exc}"
                errors.append(f"{record.npz_path}: {record.error}")
                if args.fail_fast:
                    raise

        failed_records = [record for record in run.rollout_records if record.error]
        if failed_records:
            run.processing_error = (
                f"{len(failed_records)}/{run.rollout_count} rollout(s) failed"
            )
        elif len(processed) != run.rollout_count:
            run.processing_error = (
                f"processed {len(processed)}/{run.rollout_count} rollouts"
            )
        else:
            run.stats = build_eval_stats(run, processed, args)
            write_json_atomic(run.eval_dir / "eval_stats.json", run.stats)
    return errors


def select_best_runs(runs: Sequence[EvalRun]) -> Dict[ConfigKey, EvalRun]:
    grouped: Dict[ConfigKey, list[EvalRun]] = {}
    for run in runs:
        if run.is_valid_complete:
            grouped.setdefault(run.config, []).append(run)

    selected = {}
    for config, candidates in grouped.items():
        candidates = sorted(candidates, key=lambda run: run.eval_dir.name)
        selected[config] = max(
            candidates,
            key=lambda run: (run.success_rate, run.rollout_count),
        )
        best = selected[config]
        evaluated_candidates = [
            {
                "eval_dir": str(run.eval_dir),
                "selected": run is best,
                "success_count": run.success_count,
                "failure_count": run.failure_count,
                "rollout_count": run.rollout_count,
                "success_rate": run.success_rate,
            }
            for run in candidates
        ]
        if best.stats is not None:
            best_stats = dict(best.stats)
            best_stats["selected_eval"] = best.eval_dir.name
            best_stats["evaluated_candidates"] = evaluated_candidates
            config_dir = best.eval_dir.parent.parent
            write_json_atomic(config_dir / "best_stats.json", best_stats)
    return selected


def expected_grid(args) -> set[ConfigKey]:
    masks = [normalize_mask_input(value) for value in args.mask_modes]
    grid = set()
    for task_name in args.tasks:
        for estimator in args.estimators:
            estimator_masks = [None] if estimator == "gt" else masks
            for cameras in args.camera_setups:
                for resolution in args.resolutions:
                    for mask_input in estimator_masks:
                        grid.add(ConfigKey(
                            task_name=task_name,
                            depth_estimator=estimator,
                            camera_names=tuple(cameras),
                            resolution=int(resolution),
                            mask_input=mask_input,
                        ))
    return grid


def metric_columns(metric_names):
    columns = []
    for metric_name in metric_names:
        for group_prefix in ("", "successful_", "failed_"):
            for aggregation in ("all_frames", "rollout_means"):
                for field_name in SUMMARY_FIELDS:
                    columns.append(
                        f"{metric_name}_{group_prefix}{aggregation}_{field_name}"
                    )
    return columns


def add_metric_columns(row, stats, metric_names):
    for column in metric_columns(metric_names):
        row[column] = None
    if stats is None:
        return
    for metric_name in metric_names:
        metric = stats["metrics"][metric_name]
        for group_prefix, group_key in (
            ("", None),
            ("successful_", "successful_rollouts"),
            ("failed_", "failed_rollouts"),
        ):
            group = metric if group_key is None else metric.get(group_key)
            if group is None:
                continue
            for aggregation in ("all_frames", "rollout_means"):
                summary = group[aggregation]
                for field_name in SUMMARY_FIELDS:
                    row[
                        f"{metric_name}_{group_prefix}{aggregation}_{field_name}"
                    ] = summary[field_name]


def config_identity(config: ConfigKey) -> Dict[str, Any]:
    return {
        "task_name": config.task_name,
        "depth_estimator": config.depth_estimator,
        "camera_names": config.camera_text,
        "resolution": config.resolution,
        "mask_input": config.mask_text,
    }


def status_for_config(runs: Sequence[EvalRun], selected: Optional[EvalRun]) -> str:
    if selected is not None:
        return "processing_error" if selected.processing_error else "complete"
    if not runs:
        return "not_evaluated"
    if any(run.processing_error for run in runs):
        return "processing_error"
    statuses = {run.manifest_status for run in runs}
    if "running" in statuses:
        return "running"
    if "failed" in statuses:
        return "failed"
    if any(run.is_processable for run in runs):
        return "incomplete"
    return "invalid"


def build_configuration_rows(runs, selected, args):
    by_config: Dict[ConfigKey, list[EvalRun]] = {}
    for run in runs:
        by_config.setdefault(run.config, []).append(run)
    configs = expected_grid(args) | set(by_config)
    rows = []
    for config in sorted(configs, key=ConfigKey.sort_key):
        config_runs = by_config.get(config, [])
        best = selected.get(config)
        row = config_identity(config)
        row.update({
            "status": status_for_config(config_runs, best),
            "selected_eval": best.eval_dir.name if best else None,
            "eval_directory_count": len(config_runs),
            "valid_complete_eval_count": sum(
                run.is_valid_complete for run in config_runs
            ),
            "processed_eval_count": sum(run.stats is not None for run in config_runs),
            "rollout_count": best.rollout_count if best else None,
            "success_count": best.success_count if best else None,
            "failure_count": best.failure_count if best else None,
            "success_rate": best.success_rate if best else None,
            "success_rate_percent": best.success_rate * 100.0 if best else None,
            "manifest_avg_sr": best.manifest.get("avg_sr") if best else None,
            "manifest_std_sr": best.manifest.get("std_sr") if best else None,
            "manifest_sr_list": json.dumps(best.manifest.get("sr_list"))
                if best and best.manifest.get("sr_list") is not None else None,
            "total_policy_decisions": best.stats.get("total_policy_decisions")
                if best and best.stats else None,
            "selected_eval_dir": str(best.eval_dir) if best else None,
            "best_stats_json": str(best.eval_dir.parent.parent / "best_stats.json")
                if best and best.stats else None,
            "pair_mode": args.pair_mode if best and best.stats else None,
            "norm_mode": args.norm_mode if best and best.stats else None,
            "robust_center": args.robust_center
                if best and best.stats else None,
        })
        add_metric_columns(row, best.stats if best else None, args.metrics)
        rows.append(row)
    return rows


def build_evaluation_rows(runs, selected, args):
    rows = []
    for run in sorted(runs, key=lambda item: (item.config.sort_key(), item.eval_dir.name)):
        row = config_identity(run.config)
        is_selected = selected.get(run.config) is run
        if run.is_valid_complete:
            status = "complete"
        elif run.is_processable:
            status = "incomplete"
        else:
            status = run.manifest_status
        if run.stats is not None:
            metrics_status = "processed"
        elif run.processing_error:
            metrics_status = "processing_error"
        elif run.is_processable:
            metrics_status = "not_processed"
        else:
            metrics_status = "no_paired_rollouts"
        row.update({
            "eval_name": run.eval_dir.name,
            "eval_dir": str(run.eval_dir),
            "status": status,
            "metrics_status": metrics_status,
            "is_selected": is_selected,
            "validation_error": run.validation_error,
            "processing_error": run.processing_error,
            "expected_rollouts": run.expected_rollouts,
            "artifact_pair_count": run.artifact_pair_count,
            "processed_rollout_count": len([
                record for record in run.rollout_records if record.result is not None
            ]),
            "video_error_count": sum(
                record.video_error is not None for record in run.rollout_records
            ),
            "rollout_count": run.rollout_count or None,
            "success_count": run.success_count if run.rollout_count else None,
            "failure_count": run.failure_count if run.rollout_count else None,
            "success_rate": run.success_rate,
            "success_rate_percent": run.success_rate * 100.0
                if run.success_rate is not None else None,
            "manifest_avg_sr": run.manifest.get("avg_sr"),
            "manifest_std_sr": run.manifest.get("std_sr"),
            "started_at_utc": run.manifest.get("started_at_utc"),
            "completed_at_utc": run.manifest.get("completed_at_utc"),
            "recording_code_sha256": run.manifest.get("recording_code_sha256"),
            "policy_checkpoint": run.manifest.get("policy_checkpoint"),
            "eval_stats_json": str(run.eval_dir / "eval_stats.json")
                if run.stats else None,
        })
        add_metric_columns(row, run.stats, args.metrics)
        rows.append(row)
    return rows


def build_rollout_rows(runs, selected, args):
    rows = []
    for run in sorted(runs, key=lambda item: (item.config.sort_key(), item.eval_dir.name)):
        for record in run.rollout_records:
            result = record.result
            row = config_identity(run.config)
            row.update({
                "eval_name": run.eval_dir.name,
                "is_selected_eval": selected.get(run.config) is run,
                "rollout_file": record.npz_path.name,
                "rollout_id": result.get("rollout_id") if result else None,
                "demo_iter": result.get("demo_iter") if result else None,
                "rollout_in_demo": result.get("rollout_in_demo") if result else None,
                "outcome": "success" if record.outcome else "failure",
                "success": bool(record.outcome),
                "frame_count": result.get("frame_count") if result else None,
                "processing_error": record.error,
                "video_error": record.video_error,
                "source_npz": str(record.npz_path),
                "source_video": str(
                    run.eval_dir / "live_videos" / f"{record.npz_path.stem}.mp4"
                ),
                "metric_json": str(
                    run.eval_dir / "metrics" / f"{record.npz_path.stem}.json"
                ) if result else None,
                "pointcloud_video": str(
                    run.eval_dir / "pointcloud_videos"
                    / f"{record.npz_path.stem}_pointcloud.mp4"
                ) if result and not args.skip_videos
                    and not record.video_error else None,
                "stitched_video": str(
                    run.eval_dir / "stitched_videos"
                    / f"{record.npz_path.stem}_stitched.mp4"
                ) if result and not args.skip_videos
                    and not record.video_error else None,
            })
            for metric_name in args.metrics:
                for field_name in SUMMARY_FIELDS:
                    row[f"{metric_name}_{field_name}"] = (
                        result["metrics"][metric_name]["summary"][field_name]
                        if result else None
                    )
            rows.append(row)
    return rows


def build_metric_definition_rows(args):
    rows = []
    for name in args.metrics:
        definition = METRIC_REGISTRY[name]
        rows.append({
            "metric_name": name,
            "display_name": definition.display_name,
            "description": definition.description,
            "units": definition.units,
            "required_npz_keys": "|".join(definition.required_keys),
            "pair_mode": args.pair_mode if name == "chamfer" else None,
            "norm_mode": args.norm_mode if name == "chamfer" else None,
            "robust_center": args.robust_center if name == "chamfer" else None,
            "per_rollout_json_contains_per_decision_values": True,
        })
    return rows


def ordered_headers(rows: Sequence[Dict[str, Any]], preferred: Sequence[str]):
    seen = set()
    headers = []
    for key in preferred:
        if key not in seen:
            seen.add(key)
            headers.append(key)
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                headers.append(key)
    return headers


def _column_name(index: int) -> str:
    name = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        name = chr(65 + remainder) + name
    return name


def _clean_xml_text(value: Any) -> str:
    text = str(value)
    text = "".join(
        character for character in text
        if character in "\t\n\r" or ord(character) >= 32
    )
    return escape(text)


def _xlsx_cell(reference: str, value: Any, header=False) -> str:
    if value is None:
        return ""
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, np.generic):
        value = value.item()
    style = ' s="1"' if header else ""
    if isinstance(value, bool):
        return f'<c r="{reference}" t="b"{style}><v>{int(value)}</v></c>'
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and not math.isfinite(value):
            return ""
        number_style = style or (' s="2"' if isinstance(value, float) else "")
        return f'<c r="{reference}"{number_style}><v>{value}</v></c>'
    if isinstance(value, (dict, list, tuple)):
        value = json.dumps(value, separators=(",", ":"))
    return (
        f'<c r="{reference}" t="inlineStr"{style}>'
        f'<is><t xml:space="preserve">{_clean_xml_text(value)}</t></is></c>'
    )


def _column_width(header: str) -> float:
    lowered = header.lower()
    if "path" in lowered or "_dir" in lowered or "checkpoint" in lowered:
        return 52.0
    if "error" in lowered or "description" in lowered:
        return 55.0
    if "camera" in lowered or "sha256" in lowered:
        return 34.0
    if "task" in lowered or "estimator" in lowered:
        return 22.0
    if "metric" in lowered:
        return 22.0
    return min(max(len(header) + 2, 12), 24)


def _write_sheet_xml(archive, path, headers, rows):
    if len(rows) + 1 > XLSX_MAX_ROWS:
        raise ValueError(
            f"Worksheet has {len(rows) + 1} rows; Excel limit is {XLSX_MAX_ROWS}"
        )
    last_column = _column_name(max(len(headers), 1))
    last_row = max(len(rows) + 1, 1)
    with archive.open(path, "w") as file:
        def write(text):
            file.write(text.encode("utf-8"))

        write('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>')
        write('<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">')
        write(f'<dimension ref="A1:{last_column}{last_row}"/>')
        write('<sheetViews><sheetView workbookViewId="0">')
        write('<pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/>')
        write('</sheetView></sheetViews>')
        write('<cols>')
        for index, header in enumerate(headers, start=1):
            write(
                f'<col min="{index}" max="{index}" '
                f'width="{_column_width(header):.1f}" customWidth="1"/>'
            )
        write('</cols><sheetData>')
        write('<row r="1">')
        for column_index, header in enumerate(headers, start=1):
            reference = f"{_column_name(column_index)}1"
            write(_xlsx_cell(reference, header, header=True))
        write('</row>')
        for row_index, row in enumerate(rows, start=2):
            write(f'<row r="{row_index}">')
            for column_index, header in enumerate(headers, start=1):
                value = row.get(header)
                if value is None:
                    continue
                reference = f"{_column_name(column_index)}{row_index}"
                write(_xlsx_cell(reference, value))
            write('</row>')
        write('</sheetData>')
        write(f'<autoFilter ref="A1:{last_column}{last_row}"/>')
        write('</worksheet>')


def write_xlsx_atomic(output_path: Path, sheets):
    """Write simple styled XLSX worksheets atomically without extra packages."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{output_path.stem}.",
            suffix=".xlsx",
            dir=output_path.parent,
            delete=False,
        ) as file:
            temp_path = Path(file.name)

        with zipfile.ZipFile(
            temp_path,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
        ) as archive:
            sheet_entries = []
            used_names = set()
            for index, (raw_name, headers, rows) in enumerate(sheets, start=1):
                name = re.sub(r"[\\/*?:\[\]]", "_", raw_name)[:31] or f"Sheet{index}"
                base_name = name
                suffix = 2
                while name in used_names:
                    name = f"{base_name[:28]}_{suffix}"
                    suffix += 1
                used_names.add(name)
                sheet_entries.append((name, headers, rows))

            content_overrides = "".join(
                f'<Override PartName="/xl/worksheets/sheet{index}.xml" '
                'ContentType="application/vnd.openxmlformats-officedocument.'
                'spreadsheetml.worksheet+xml"/>'
                for index in range(1, len(sheet_entries) + 1)
            )
            archive.writestr(
                "[Content_Types].xml",
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                '<Default Extension="xml" ContentType="application/xml"/>'
                '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
                '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
                f'{content_overrides}</Types>',
            )
            archive.writestr(
                "_rels/.rels",
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
                '</Relationships>',
            )
            workbook_sheets = "".join(
                f'<sheet name="{_clean_xml_text(name)}" sheetId="{index}" r:id="rId{index}"/>'
                for index, (name, _, _) in enumerate(sheet_entries, start=1)
            )
            archive.writestr(
                "xl/workbook.xml",
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
                'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
                f'<sheets>{workbook_sheets}</sheets></workbook>',
            )
            workbook_rels = "".join(
                f'<Relationship Id="rId{index}" '
                'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
                f'Target="worksheets/sheet{index}.xml"/>'
                for index in range(1, len(sheet_entries) + 1)
            )
            styles_id = len(sheet_entries) + 1
            workbook_rels += (
                f'<Relationship Id="rId{styles_id}" '
                'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
                'Target="styles.xml"/>'
            )
            archive.writestr(
                "xl/_rels/workbook.xml.rels",
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                f'{workbook_rels}</Relationships>',
            )
            archive.writestr(
                "xl/styles.xml",
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
                '<numFmts count="1"><numFmt numFmtId="164" formatCode="0.000000"/></numFmts>'
                '<fonts count="2"><font><sz val="11"/><name val="Calibri"/></font>'
                '<font><b/><color rgb="FFFFFFFF"/><sz val="11"/><name val="Calibri"/></font></fonts>'
                '<fills count="3"><fill><patternFill patternType="none"/></fill>'
                '<fill><patternFill patternType="gray125"/></fill>'
                '<fill><patternFill patternType="solid"><fgColor rgb="FF4472C4"/><bgColor indexed="64"/></patternFill></fill></fills>'
                '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>'
                '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
                '<cellXfs count="3"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>'
                '<xf numFmtId="0" fontId="1" fillId="2" borderId="0" xfId="0" applyFont="1" applyFill="1" applyAlignment="1"><alignment horizontal="center"/></xf>'
                '<xf numFmtId="164" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"/></cellXfs>'
                '<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>'
                '</styleSheet>',
            )
            for index, (_, headers, rows) in enumerate(sheet_entries, start=1):
                _write_sheet_xml(
                    archive,
                    f"xl/worksheets/sheet{index}.xml",
                    headers,
                    rows,
                )

        os.replace(temp_path, output_path)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def build_workbook(input_root, runs, selected, args, output_path):
    configuration_rows = build_configuration_rows(runs, selected, args)
    evaluation_rows = build_evaluation_rows(runs, selected, args)
    rollout_rows = build_rollout_rows(runs, selected, args)
    metric_rows = build_metric_definition_rows(args)

    identity = (
        "task_name", "depth_estimator", "camera_names", "resolution", "mask_input"
    )
    configuration_headers = ordered_headers(
        configuration_rows,
        identity + (
            "status", "selected_eval", "eval_directory_count",
            "valid_complete_eval_count", "processed_eval_count",
            "rollout_count", "success_count", "failure_count", "success_rate",
            "success_rate_percent",
            "manifest_avg_sr", "manifest_std_sr", "manifest_sr_list",
            "total_policy_decisions", "selected_eval_dir", "best_stats_json",
            "pair_mode", "norm_mode", "robust_center",
        ) + tuple(metric_columns(args.metrics)),
    )
    evaluation_headers = ordered_headers(
        evaluation_rows,
        identity + (
            "eval_name", "status", "metrics_status", "is_selected",
            "validation_error",
            "processing_error", "expected_rollouts", "artifact_pair_count",
            "processed_rollout_count", "video_error_count", "rollout_count",
            "success_count",
            "failure_count", "success_rate", "manifest_avg_sr",
            "success_rate_percent",
            "manifest_std_sr", "started_at_utc", "completed_at_utc",
            "eval_dir", "eval_stats_json",
        ) + tuple(metric_columns(args.metrics)),
    )
    rollout_metric_headers = tuple(
        f"{name}_{field_name}"
        for name in args.metrics
        for field_name in SUMMARY_FIELDS
    )
    rollout_headers = ordered_headers(
        rollout_rows,
        identity + (
            "eval_name", "is_selected_eval", "rollout_file", "rollout_id",
            "demo_iter", "rollout_in_demo", "outcome", "success",
            "frame_count", "processing_error", "video_error",
        ) + rollout_metric_headers + (
            "source_npz", "source_video", "metric_json",
            "pointcloud_video", "stitched_video",
        ),
    )
    metric_headers = ordered_headers(
        metric_rows,
        (
            "metric_name", "display_name", "description", "units",
            "required_npz_keys", "pair_mode", "norm_mode", "robust_center",
            "per_rollout_json_contains_per_decision_values",
        ),
    )
    write_xlsx_atomic(output_path, [
        ("Configurations", configuration_headers, configuration_rows),
        ("Evaluations", evaluation_headers, evaluation_rows),
        ("Rollouts", rollout_headers, rollout_rows),
        ("Metric Definitions", metric_headers, metric_rows),
    ])
    return {
        "input_root": str(input_root),
        "workbook": str(output_path),
        "configuration_rows": len(configuration_rows),
        "discovered_evaluations": len(runs),
        "processed_evaluations": sum(run.stats is not None for run in runs),
        "processed_rollouts": len([
            row for row in rollout_rows if row["processing_error"] is None
        ]),
        "rollout_errors": len([
            row for row in rollout_rows if row["processing_error"] is not None
        ]),
        "video_errors": len([
            row for row in rollout_rows if row["video_error"] is not None
        ]),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Process every paired Instant Policy rollout artifact and "
            "maintain a full-grid Excel workbook."
        )
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="IP artifact root containing configuration rollouts/eval_0000 dirs.",
    )
    parser.add_argument(
        "--output_xlsx",
        type=Path,
        default=None,
        help="Workbook path (default: <input_dir>/ip_rollout_results.xlsx).",
    )
    parser.add_argument("--pattern", default="rollout_*.npz")
    parser.add_argument(
        "--expected_rollouts",
        type=int,
        default=200,
        help="Expected paired rollout IDs per configuration (default: 200).",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=sorted(METRIC_REGISTRY),
        default=list(DEFAULT_METRICS),
        help=(
            "Metrics to calculate (default: Chamfer, directional distances, "
            "and precision/recall/F-score at 5/10/20 mm)."
        ),
    )
    parser.add_argument(
        "--pair_mode",
        choices=("baseline", "independent"),
        default="baseline",
    )
    parser.add_argument(
        "--norm_mode",
        choices=("unit_sphere", "unit_cube", "std"),
        default="unit_sphere",
    )
    parser.add_argument("--robust_center", action="store_true")
    parser.add_argument(
        "--skip_videos",
        action="store_true",
        help="Calculate metrics/tables without rendering or stitching videos.",
    )
    parser.add_argument(
        "--fail_fast",
        action="store_true",
        help="Stop at the first invalid NPZ/metric instead of recording all errors.",
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--render_width", type=int, default=900)
    parser.add_argument("--render_height", type=int, default=700)
    parser.add_argument("--point_size", type=float, default=2.0)
    parser.add_argument("--agent_axis_size", type=float, default=0.08)
    parser.add_argument("--elevation", type=float, default=25.0)
    parser.add_argument("--azimuth", type=float, default=-60.0)
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument(
        "--estimators",
        nargs="+",
        default=list(DEFAULT_ESTIMATORS),
    )
    parser.add_argument(
        "--camera_setups",
        nargs="+",
        type=parse_camera_setup,
        default=list(DEFAULT_CAMERA_SETUPS),
        metavar="CAM1+CAM2",
    )
    parser.add_argument(
        "--resolutions",
        nargs="+",
        type=int,
        default=list(DEFAULT_RESOLUTIONS),
    )
    parser.add_argument(
        "--mask_modes",
        nargs="+",
        choices=("unmasked", "masked"),
        default=["unmasked", "masked"],
        help="Expected learned-estimator mask modes; GT always uses none.",
    )
    return parser.parse_args(argv)


def run_processor(args):
    input_root = args.input_dir.resolve()
    if not input_root.is_dir():
        raise NotADirectoryError(input_root)
    output_path = (
        args.output_xlsx.resolve()
        if args.output_xlsx is not None
        else input_root / "ip_rollout_results.xlsx"
    )
    if args.expected_rollouts <= 0:
        raise ValueError("--expected_rollouts must be positive")
    runs = discover_eval_runs(
        input_root,
        args.pattern,
        expected_rollouts=args.expected_rollouts,
    )
    errors = process_all_eval_runs(runs, args)
    selected = select_best_runs(runs)
    summary = build_workbook(input_root, runs, selected, args, output_path)
    summary["processing_errors"] = errors
    write_json_atomic(
        output_path.with_name(f"{output_path.stem}_summary.json"),
        summary,
    )
    return summary


def main(argv=None):
    args = parse_args(argv)
    summary = run_processor(args)
    print(
        "Processed "
        f"{summary['processed_rollouts']} rollouts from "
        f"{summary['processed_evaluations']} canonical evaluations."
    )
    print(
        f"Workbook contains {summary['configuration_rows']} configuration rows: "
        f"{summary['workbook']}"
    )
    if summary["rollout_errors"]:
        print(f"Rollout processing errors: {summary['rollout_errors']}")
    if summary["video_errors"]:
        print(f"Optional video rendering errors: {summary['video_errors']}")


if __name__ == "__main__":
    main()
