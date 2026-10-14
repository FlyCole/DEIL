#!/usr/bin/env python3
"""Select, render, score, and stitch RLBench rollout evaluations.

The input may be one ``rollouts`` directory, one run directory containing a
``rollouts`` directory, or a task root containing multiple ``<run>/rollouts``
directories. Aggregate ``best_stats.json`` files are kept both beside each
run and in a portable, Git-friendly stats tree under ``data_analysis/dp3``.
"""

import argparse
import json
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Sequence

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as Rot
from tqdm import tqdm


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATS_DIR = REPOSITORY_ROOT / "data_analysis" / "dp3" / "stats"
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from chamfer_distance_eval import (
    chamfer_kdtree,
    directional_nearest_neighbor_distances,
    extract_xyz,
    normalize_pair,
    summarize,
)


@dataclass(frozen=True)
class MetricDefinition:
    name: str
    display_name: str
    required_keys: Sequence[str]
    compute: Callable[
        [Dict[str, np.ndarray], argparse.Namespace, Dict[Any, Any]],
        np.ndarray,
    ]
    description: str
    units: str
    sample_kind: str


@dataclass(frozen=True)
class EvalCandidate:
    eval_dir: Path
    rollout_paths: Sequence[Path]
    outcomes: Sequence[bool]

    @property
    def success_count(self):
        return sum(self.outcomes)

    @property
    def rollout_count(self):
        return len(self.outcomes)

    @property
    def failure_count(self):
        return self.rollout_count - self.success_count

    @property
    def success_rate(self):
        return self.success_count / self.rollout_count


METRIC_REGISTRY: Dict[str, MetricDefinition] = {}


def register_metric(
    name,
    display_name,
    required_keys,
    description,
    units="unitless",
    sample_kind="frame",
):
    """Decorator used to add metrics without changing the video pipeline."""
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
            sample_kind=sample_kind,
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
    """Return cached predicted-to-GT and GT-to-predicted distances in meters."""
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


def _uniform_point_subset(points, maximum_point_count):
    """Select a deterministic, approximately uniform subset by array index."""
    points = np.asarray(points, dtype=np.float64)
    if maximum_point_count <= 0 or points.shape[0] <= maximum_point_count:
        return points
    indices = np.linspace(
        0,
        points.shape[0] - 1,
        num=maximum_point_count,
        dtype=np.int64,
    )
    return points[indices]


def _symmetric_nearest_neighbor_motion(source, target):
    """Return the symmetric mean nearest-neighbor displacement in meters."""
    source_to_target, target_to_source = (
        directional_nearest_neighbor_distances(source, target)
    )
    return 0.5 * (
        float(source_to_target.mean()) + float(target_to_source.mean())
    )


@register_metric(
    name="temporal_inconsistency_mm",
    display_name="Temporal inconsistency",
    required_keys=("agent_pos", "obs_pointcloud", "gt_pointcloud"),
    description=(
        "Absolute GT-compensated error in symmetric inter-frame point-cloud "
        "motion. Predicted and GT clouds are transformed to the world frame; "
        "for every consecutive pair, the absolute difference between their "
        "symmetric mean nearest-neighbor displacements is reported. Lower "
        "values indicate more temporally consistent geometry."
    ),
    units="mm",
    sample_kind="transition",
)
def compute_temporal_inconsistency_mm(data, args, cache):
    observed_world, _ = point_clouds_in_world(data, "obs_pointcloud")
    reference_world, _ = point_clouds_in_world(data, "gt_pointcloud")
    if observed_world.shape[0] != reference_world.shape[0]:
        raise ValueError(
            "Temporal frame-count mismatch: "
            f"observed={observed_world.shape[0]}, "
            f"reference={reference_world.shape[0]}"
        )
    if observed_world.shape[0] < 2:
        raise ValueError(
            "Temporal inconsistency requires at least two rollout frames"
        )

    maximum_point_count = int(getattr(args, "temporal_max_points", 1024))
    if maximum_point_count < 0:
        raise ValueError("--temporal-max-points must be non-negative")

    values = []
    for frame_index in tqdm(
        range(1, observed_world.shape[0]),
        desc="Computing temporal consistency",
        leave=False,
    ):
        predicted_previous = _uniform_point_subset(
            observed_world[frame_index - 1],
            maximum_point_count,
        )
        predicted_current = _uniform_point_subset(
            observed_world[frame_index],
            maximum_point_count,
        )
        gt_previous = _uniform_point_subset(
            reference_world[frame_index - 1],
            maximum_point_count,
        )
        gt_current = _uniform_point_subset(
            reference_world[frame_index],
            maximum_point_count,
        )
        predicted_motion = _symmetric_nearest_neighbor_motion(
            predicted_previous,
            predicted_current,
        )
        gt_motion = _symmetric_nearest_neighbor_motion(
            gt_previous,
            gt_current,
        )
        values.append(abs(predicted_motion - gt_motion) * 1000.0)
    return np.asarray(values, dtype=np.float64)


DEFAULT_METRICS = tuple(METRIC_REGISTRY)
VIDEO_OVERLAY_SCHEMA_VERSION = 1
VIDEO_OVERLAY_METRICS = (
    "chamfer",
    "accuracy_mm",
    "completeness_mm",
    "fscore_20mm",
)
VIDEO_OVERLAY_LABELS = {
    "chamfer": "Chamfer distance",
    "accuracy_mm": "Accuracy",
    "completeness_mm": "Completeness",
    "fscore_20mm": "F-score @ 20 mm",
}
HYDRA_RUN_PATTERN = re.compile(
    r"^rlb_(?P<task>.+)-dp3-(?P<tag>[^-]+)-"
    r"(?P<estimator>.+)_seed(?P<seed>\d+)$"
)


def load_rollout_data(npz_path):
    with np.load(npz_path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}

    required = ("agent_pos", "obs_pointcloud")
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"{npz_path} is missing required keys: {missing}")

    agent_pos = np.asarray(data["agent_pos"])
    point_cloud = np.asarray(data["obs_pointcloud"])
    if agent_pos.ndim != 2 or agent_pos.shape[1] < 7:
        raise ValueError(f"agent_pos must have shape (T, D>=7), got {agent_pos.shape}")
    if point_cloud.ndim != 3 or point_cloud.shape[-1] < 3:
        raise ValueError(
            "obs_pointcloud must have shape (T, N, D>=3), "
            f"got {point_cloud.shape}"
        )
    if agent_pos.shape[0] != point_cloud.shape[0]:
        raise ValueError(
            "Rollout frame-count mismatch: "
            f"agent_pos={agent_pos.shape[0]}, point_cloud={point_cloud.shape[0]}"
        )
    if not np.isfinite(agent_pos).all() or not np.isfinite(point_cloud).all():
        raise ValueError(f"{npz_path} contains non-finite rollout data")
    return data


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
        expected_count = (
            frame_count - 1
            if definition.sample_kind == "transition"
            else frame_count
        )
        if values.shape != (expected_count,):
            raise ValueError(
                f"Metric {name!r} returned shape {values.shape}; "
                f"expected ({expected_count},) for "
                f"{definition.sample_kind} samples"
            )
        if not np.isfinite(values).all():
            raise ValueError(f"Metric {name!r} returned non-finite values")
        results[name] = {
            "display_name": definition.display_name,
            "description": definition.description,
            "units": definition.units,
            "sample_kind": definition.sample_kind,
            "values": values,
            "summary": summarize(values),
        }
    return results


def rollout_outcome(npz_path):
    """Read a rollout's success label from its *_success/failure.npz name."""
    outcome = npz_path.stem.rsplit("_", maxsplit=1)[-1].lower()
    if outcome == "success":
        return True
    if outcome == "failure":
        return False
    raise ValueError(
        f"Cannot determine rollout outcome from {npz_path.name!r}; expected "
        "the filename to end in '_success.npz' or '_failure.npz'."
    )


def find_eval_candidates(rollouts_dir, pattern):
    candidates = []
    eval_dirs = sorted(
        path for path in rollouts_dir.glob("eval_*") if path.is_dir()
    )
    if not eval_dirs:
        raise FileNotFoundError(
            f"No eval_* directories found in {rollouts_dir}"
        )

    for eval_dir in eval_dirs:
        state_dir = eval_dir / "live_states"
        rollout_paths = tuple(sorted(state_dir.glob(pattern))) \
            if state_dir.is_dir() else ()
        if not rollout_paths:
            warnings.warn(
                f"Skipping {eval_dir}: no rollout data matching {pattern!r} "
                f"in {state_dir}",
                stacklevel=2,
            )
            continue
        outcomes = tuple(rollout_outcome(path) for path in rollout_paths)
        candidates.append(EvalCandidate(eval_dir, rollout_paths, outcomes))

    if not candidates:
        raise FileNotFoundError(
            f"No rollout data matching {pattern!r} in any eval_* directory "
            f"under {rollouts_dir}"
        )
    return candidates


def discover_rollouts_dirs(input_dir):
    """Resolve supported input layouts to one or more rollout roots.

    Supported layouts are:

    * ``<rollouts>/eval_*`` (the original interface),
    * ``<run>/rollouts/eval_*``, and
    * ``<task-root>/<run>/rollouts/eval_*``.

    Task-root discovery intentionally examines immediate run subdirectories so
    unrelated nested analysis/output folders are not processed accidentally.
    """
    input_dir = Path(input_dir).resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(input_dir)

    contains_eval_dirs = any(
        path.is_dir() for path in input_dir.glob("eval_*")
    )
    if contains_eval_dirs:
        return (input_dir,)

    direct_rollouts_dir = input_dir / "rollouts"
    if direct_rollouts_dir.is_dir():
        return (direct_rollouts_dir.resolve(),)

    rollout_dirs = tuple(sorted(
        (run_dir / "rollouts").resolve()
        for run_dir in input_dir.iterdir()
        if run_dir.is_dir() and (run_dir / "rollouts").is_dir()
    ))
    if rollout_dirs:
        return rollout_dirs

    raise FileNotFoundError(
        f"Could not find rollout data under {input_dir}. Expected eval_* "
        "directories directly, a rollouts/ directory, or immediate "
        "<run>/rollouts/ subdirectories."
    )


def select_best_eval(candidates):
    """Choose by success rate, then sample count, then earliest eval name."""
    candidates_by_name = sorted(
        candidates,
        key=lambda candidate: candidate.eval_dir.name,
    )
    return max(
        candidates_by_name,
        key=lambda candidate: (
            candidate.success_rate,
            candidate.rollout_count,
        ),
    )


def pose_transform(agent_pos):
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rot.from_rotvec(agent_pos[3:6]).as_matrix()
    transform[:3, 3] = agent_pos[:3]
    return transform


def point_clouds_in_world(data, point_cloud_key="obs_pointcloud"):
    point_clouds = extract_xyz(data[point_cloud_key]).astype(np.float64)
    frame_name = "eef"
    if "point_cloud_frame" in data:
        frame_name = str(np.asarray(data["point_cloud_frame"]).item())
    if frame_name not in ("eef", "world"):
        raise ValueError(f"Unsupported point_cloud_frame: {frame_name!r}")

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
    return [(center[axis] - half_span, center[axis] + half_span) for axis in range(3)]


def metric_overlay_text(metric_results, frame_idx):
    lines = []
    for metric_name in VIDEO_OVERLAY_METRICS:
        result = metric_results.get(metric_name)
        if result is None:
            continue
        value = float(result["values"][frame_idx])
        if result["units"] == "fraction":
            formatted_value = f"{value * 100.0:.1f}%"
        elif result["units"] == "mm":
            formatted_value = f"{value:.2f} mm"
        else:
            formatted_value = f"{value:.6f}"
        lines.append(
            f"{VIDEO_OVERLAY_LABELS[metric_name]}: {formatted_value}"
        )
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
        for frame_idx in tqdm(range(frame_count), desc="Rendering point clouds", leave=False):
            axis.clear()
            points = world_clouds[frame_idx]
            transform = transforms[frame_idx]
            state = data["agent_pos"][frame_idx]

            axis.scatter(
                points[:, 0],
                points[:, 1],
                points[:, 2],
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
    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)


def stitch_videos(rgb_path, diagnostic_path, output_path):
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

    writer = None
    written_frames = 0
    try:
        while True:
            rgb_ok, rgb_frame = rgb_capture.read()
            diagnostic_ok, diagnostic_frame = diagnostic_capture.read()
            if not rgb_ok or not diagnostic_ok:
                break

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

    if rgb_frames != diagnostic_frames:
        warnings.warn(
            "Video frame-count mismatch; stitched the shared prefix only: "
            f"rgb={rgb_frames}, diagnostic={diagnostic_frames}",
            stacklevel=2,
        )
    return written_frames


def json_metric_results(npz_path, data, metric_results, args, outcome=None):
    if outcome is None:
        outcome = rollout_outcome(npz_path)
    metrics = {}
    for name, result in metric_results.items():
        sample_kind = result["sample_kind"]
        sample_field = (
            "per_transition" if sample_kind == "transition" else "per_frame"
        )
        metrics[name] = {
            "display_name": result["display_name"],
            "description": result["description"],
            "units": result["units"],
            "sample_kind": sample_kind,
            "summary": result["summary"],
            sample_field: result["values"].tolist(),
        }
        if name == "chamfer":
            metrics[name]["normalization"] = {
                "pair_mode": args.pair_mode,
                "norm_mode": args.norm_mode,
                "robust_center": args.robust_center,
            }
        if name == "temporal_inconsistency_mm":
            metrics[name]["configuration"] = {
                "coordinate_frame": "world",
                "max_points": int(args.temporal_max_points),
                "motion_metric": "symmetric_mean_nearest_neighbor_l2",
                "gt_compensation": "absolute_motion_difference",
            }
    return {
        "source_npz": str(npz_path),
        "success": outcome,
        "outcome": "success" if outcome else "failure",
        "frame_count": int(data["agent_pos"].shape[0]),
        "video_overlay": None if getattr(args, "skip_videos", False) else {
            "schema_version": VIDEO_OVERLAY_SCHEMA_VERSION,
            "metrics": [
                name for name in VIDEO_OVERLAY_METRICS
                if name in metric_results
            ],
        },
        "metrics": metrics,
    }


def rollout_artifact_paths(npz_path, eval_dir):
    """Return metric, diagnostic, RGB, and stitched paths for one rollout."""
    stem = npz_path.stem
    return (
        eval_dir / "metrics" / f"{stem}.json",
        eval_dir / "pointcloud_videos" / f"{stem}_pointcloud.mp4",
        eval_dir / "live_videos" / f"{stem}.mp4",
        eval_dir / "stitched_videos" / f"{stem}_stitched.mp4",
    )


def video_is_complete(path, minimum_frame_count):
    """Check that a video opens and its final expected frame can be read."""
    try:
        missing_or_empty = not path.is_file() or path.stat().st_size == 0
    except OSError:
        return False
    if minimum_frame_count <= 0 or missing_or_empty:
        return False
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return False
        reported_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        if reported_count < minimum_frame_count:
            return False
        capture.set(cv2.CAP_PROP_POS_FRAMES, minimum_frame_count - 1)
        readable, _ = capture.read()
        return bool(readable)
    finally:
        capture.release()


def rollout_json_is_compatible(payload, args, outcome):
    """Validate a saved rollout record against the requested metric setup."""
    if not isinstance(payload, dict):
        return False
    if payload.get("success") is not outcome:
        return False
    try:
        frame_count = int(payload["frame_count"])
    except (KeyError, TypeError, ValueError):
        return False
    if frame_count <= 0:
        return False

    if not getattr(args, "skip_videos", False):
        expected_video_overlay = {
            "schema_version": VIDEO_OVERLAY_SCHEMA_VERSION,
            "metrics": [
                name for name in VIDEO_OVERLAY_METRICS
                if name in args.metrics
            ],
        }
        if payload.get("video_overlay") != expected_video_overlay:
            return False

    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        return False
    for metric_name in args.metrics:
        definition = METRIC_REGISTRY[metric_name]
        metric = metrics.get(metric_name)
        if not isinstance(metric, dict):
            return False
        summary = metric.get("summary")
        if not isinstance(summary, dict) or "mean" not in summary:
            return False
        try:
            mean = float(summary["mean"])
            sample_field = (
                "per_transition"
                if definition.sample_kind == "transition"
                else "per_frame"
            )
            values = np.asarray(metric[sample_field], dtype=np.float64)
        except (KeyError, TypeError, ValueError):
            return False
        expected_count = (
            frame_count - 1
            if definition.sample_kind == "transition"
            else frame_count
        )
        if (
            metric.get("sample_kind", "frame") != definition.sample_kind
            or values.shape != (expected_count,)
            or not np.isfinite(values).all()
        ):
            return False
        if not np.isfinite(mean):
            return False

    if "chamfer" in args.metrics:
        normalization = metrics["chamfer"].get("normalization")
        expected_normalization = {
            "pair_mode": args.pair_mode,
            "norm_mode": args.norm_mode,
            "robust_center": args.robust_center,
        }
        if normalization != expected_normalization:
            return False
    if "temporal_inconsistency_mm" in args.metrics:
        configuration = metrics["temporal_inconsistency_mm"].get(
            "configuration"
        )
        expected_configuration = {
            "coordinate_frame": "world",
            "max_points": int(args.temporal_max_points),
            "motion_metric": "symmetric_mean_nearest_neighbor_l2",
            "gt_compensation": "absolute_motion_difference",
        }
        if configuration != expected_configuration:
            return False
    return True


def load_completed_rollout(npz_path, eval_dir, args, outcome):
    """Load one resumable rollout, or return ``None`` if any artifact is stale."""
    metric_path, diagnostic_path, rgb_path, stitched_path = (
        rollout_artifact_paths(npz_path, eval_dir)
    )
    if not metric_path.is_file():
        return None
    try:
        with metric_path.open(encoding="utf-8") as file:
            payload = json.load(file)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    if not rollout_json_is_compatible(payload, args, outcome):
        return None

    frame_count = int(payload["frame_count"])
    if not getattr(args, "skip_videos", False):
        if not video_is_complete(diagnostic_path, frame_count):
            return None
        if rgb_path.is_file():
            rgb_capture = cv2.VideoCapture(str(rgb_path))
            try:
                if not rgb_capture.isOpened():
                    return None
                reported_rgb_frames = rgb_capture.get(cv2.CAP_PROP_FRAME_COUNT)
                if not np.isfinite(reported_rgb_frames):
                    return None
                rgb_frame_count = int(round(reported_rgb_frames))
            finally:
                rgb_capture.release()
            synchronized_frames = min(frame_count, rgb_frame_count)
            if not video_is_complete(stitched_path, synchronized_frames):
                return None

    # Ignore metrics saved by an older run but not requested in this run.
    payload["metrics"] = {
        name: payload["metrics"][name] for name in args.metrics
    }
    return payload


def process_rollout(npz_path, eval_dir, args, outcome=None):
    if outcome is None:
        outcome = rollout_outcome(npz_path)
    data = load_rollout_data(npz_path)
    metric_results = compute_metrics(data, args.metrics, args)
    fps = float(np.asarray(data.get("fps", 10.0)).item())
    metric_path, diagnostic_path, rgb_path, stitched_path = (
        rollout_artifact_paths(npz_path, eval_dir)
    )

    if not args.skip_videos:
        # Invalidate old derived videos before publishing the new metric record.
        # If interrupted, --resume will retry rather than pair new metrics with
        # an old or partial video.
        diagnostic_path.unlink(missing_ok=True)
        stitched_path.unlink(missing_ok=True)

    json_results = json_metric_results(
        npz_path,
        data,
        metric_results,
        args,
        outcome=outcome,
    )
    metric_path.parent.mkdir(parents=True, exist_ok=True)
    with metric_path.open("w", encoding="utf-8") as file:
        json.dump(json_results, file, indent=2)

    if not args.skip_videos:
        render_point_cloud_video(
            data,
            metric_results,
            output_path=diagnostic_path,
            args=args,
            fps=fps,
        )

        if rgb_path.is_file():
            stitch_videos(rgb_path, diagnostic_path, stitched_path)
        else:
            warnings.warn(
                f"No matching RGB rollout video for {npz_path.name}; "
                "skipped stitching.",
                stacklevel=2,
            )
    return json_results


def summarize_metric_group(processed_rollouts, metric_name):
    definition = METRIC_REGISTRY[metric_name]
    sample_field = (
        "per_transition"
        if definition.sample_kind == "transition"
        else "per_frame"
    )
    sample_values = np.concatenate([
        np.asarray(
            rollout["metrics"][metric_name][sample_field],
            dtype=np.float64,
        )
        for rollout in processed_rollouts
    ])
    rollout_means = np.asarray([
        rollout["metrics"][metric_name]["summary"]["mean"]
        for rollout in processed_rollouts
    ], dtype=np.float64)
    sample_summary_name = (
        "all_transitions"
        if definition.sample_kind == "transition"
        else "all_frames"
    )
    return {
        sample_summary_name: summarize(sample_values),
        "rollout_means": summarize(rollout_means),
    }


def run_metadata(rollouts_dir):
    """Collect metadata that remains usable after stats are copied elsewhere."""
    run_dir = rollouts_dir.parent
    metadata = {
        "run_name": run_dir.name,
        "task_group": run_dir.parent.name,
    }
    match = HYDRA_RUN_PATTERN.fullmatch(run_dir.name)
    if match is not None:
        metadata.update({
            "task_name": match.group("task"),
            "depth_estimator": match.group("estimator"),
            "seed": int(match.group("seed")),
        })

    config_path = run_dir / ".hydra" / "config.yaml"
    if config_path.is_file():
        try:
            with config_path.open(encoding="utf-8") as file:
                config = yaml.safe_load(file) or {}
        except (OSError, ValueError, TypeError, yaml.YAMLError) as exc:
            warnings.warn(
                f"Could not load run metadata from {config_path}: {exc}",
                stacklevel=2,
            )
        else:
            task_config = (
                config.get("task") if isinstance(config, dict) else None
            )
            task_config = (
                task_config if isinstance(task_config, dict) else {}
            )
            runner = task_config.get("env_runner")
            runner = runner if isinstance(runner, dict) else {}
            config_metadata = {
                "task_name": task_config.get("task_name"),
                "depth_estimator": task_config.get("depth_est_name"),
                "camera_names": runner.get("camera_names"),
                "resolution": runner.get("resolution"),
            }
            metadata.update({
                key: value
                for key, value in config_metadata.items()
                if value is not None
            })

    metadata.setdefault("task_name", run_dir.parent.name)
    metadata.setdefault("depth_estimator", "unknown_estimator")
    metadata.setdefault("camera_names", [])
    metadata.setdefault("resolution", 0)
    metadata.setdefault("seed", 0)
    camera_names = metadata["camera_names"]
    if isinstance(camera_names, str):
        camera_names = [camera_names]
    metadata["camera_names"] = [str(name) for name in camera_names]
    try:
        metadata["resolution"] = int(metadata["resolution"] or 0)
    except (TypeError, ValueError):
        metadata["resolution"] = 0
    return metadata


def build_best_stats(rollouts_dir, best_eval, candidates, processed_rollouts, args):
    metrics = {}
    for metric_name in args.metrics:
        definition = METRIC_REGISTRY[metric_name]
        metric_stats = {
            "display_name": definition.display_name,
            "description": definition.description,
            "units": definition.units,
            "sample_kind": definition.sample_kind,
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
        if metric_name == "temporal_inconsistency_mm":
            metric_stats["configuration"] = {
                "coordinate_frame": "world",
                "max_points": int(args.temporal_max_points),
                "motion_metric": "symmetric_mean_nearest_neighbor_l2",
                "gt_compensation": "absolute_motion_difference",
            }
        metrics[metric_name] = metric_stats

    rollouts = []
    for result in processed_rollouts:
        rollouts.append({
            "source_npz": result["source_npz"],
            "success": result["success"],
            "outcome": result["outcome"],
            "frame_count": result["frame_count"],
            "metrics": {
                name: metric["summary"]
                for name, metric in result["metrics"].items()
            },
        })

    evaluated = []
    for candidate in candidates:
        evaluated.append({
            "eval_dir": str(candidate.eval_dir),
            "selected": candidate.eval_dir == best_eval.eval_dir,
            "success_count": candidate.success_count,
            "failure_count": candidate.failure_count,
            "rollout_count": candidate.rollout_count,
            "success_rate": candidate.success_rate,
            "success_rate_percent": candidate.success_rate * 100.0,
        })

    return {
        "schema_version": 2,
        **run_metadata(rollouts_dir),
        "rollouts_dir": str(rollouts_dir),
        "best_eval_dir": str(best_eval.eval_dir),
        "best_eval_name": best_eval.eval_dir.name,
        "success_count": best_eval.success_count,
        "failure_count": best_eval.failure_count,
        "rollout_count": best_eval.rollout_count,
        "best_success_rate": best_eval.success_rate,
        "best_success_rate_percent": best_eval.success_rate * 100.0,
        "success_rate": best_eval.success_rate,
        "success_rate_percent": best_eval.success_rate * 100.0,
        "metrics": metrics,
        "rollouts": rollouts,
        "evaluated_candidates": evaluated,
    }


def portable_stats_path(rollouts_dir, stats_dir):
    """Return the portable mirror path for a run's aggregate statistics."""
    run_dir = rollouts_dir.parent
    return (
        stats_dir.resolve()
        / run_dir.parent.name
        / run_dir.name
        / "best_stats.json"
    )


def save_best_stats(rollouts_dir, best_eval, candidates, processed_rollouts, args):
    output_path = rollouts_dir.parent / "best_stats.json"
    stats = build_best_stats(
        rollouts_dir,
        best_eval,
        candidates,
        processed_rollouts,
        args,
    )
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(stats, file, indent=2)

    portable_path = portable_stats_path(rollouts_dir, args.stats_dir)
    portable_path.parent.mkdir(parents=True, exist_ok=True)
    with portable_path.open("w", encoding="utf-8") as file:
        json.dump(stats, file, indent=2)
    return output_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Compute rollout metrics, render white-background point-cloud videos, "
            "and stitch them beside recorded RLBench RGB videos."
        )
    )
    parser.add_argument(
        "--input-dir",
        "--input_dir",
        dest="input_dir",
        type=Path,
        required=True,
        help=(
            "A rollouts root containing eval_* directories, a run directory "
            "containing rollouts/, or a task root containing multiple "
            "<run>/rollouts/ directories. For every discovered run, only the "
            "eval with the highest success rate is processed and "
            "best_stats.json is written beside its rollouts directory."
        ),
    )
    parser.add_argument(
        "--stats-dir",
        "--stats_dir",
        dest="stats_dir",
        type=Path,
        default=DEFAULT_STATS_DIR,
        help=(
            "Portable aggregate-statistics root. Each run is copied to "
            "<stats-dir>/<task>/<run>/best_stats.json (default: "
            f"{DEFAULT_STATS_DIR})."
        ),
    )
    parser.add_argument("--pattern", default="rollout_*.npz")
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Reuse complete per-rollout metric/video artifacts and process "
            "only missing, corrupt, or incompatible rollouts."
        ),
    )
    parser.add_argument(
        "--skip-videos",
        "--skip_videos",
        dest="skip_videos",
        action="store_true",
        help=(
            "Calculate metrics and summary files without rendering point-cloud "
            "videos or stitching them with RGB videos."
        ),
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=sorted(METRIC_REGISTRY),
        default=list(DEFAULT_METRICS),
        help=(
            "Metrics to calculate (default: Chamfer, directional distances, "
            "precision/recall/F-score at 5/10/20 mm, and GT-compensated "
            "temporal inconsistency)."
        ),
    )
    parser.add_argument("--pair_mode", choices=("baseline", "independent"), default="baseline")
    parser.add_argument(
        "--norm_mode",
        choices=("unit_sphere", "unit_cube", "std"),
        default="unit_sphere",
    )
    parser.add_argument("--robust_center", action="store_true")
    parser.add_argument(
        "--temporal-max-points",
        "--temporal_max_points",
        dest="temporal_max_points",
        type=int,
        default=1024,
        help=(
            "Maximum points sampled deterministically from each cloud for "
            "temporal-consistency transitions; 0 uses all points "
            "(default: 1024)."
        ),
    )
    parser.add_argument("--render_width", type=int, default=900)
    parser.add_argument("--render_height", type=int, default=700)
    parser.add_argument("--point_size", type=float, default=2.0)
    parser.add_argument("--agent_axis_size", type=float, default=0.08)
    parser.add_argument("--elevation", type=float, default=25.0)
    parser.add_argument("--azimuth", type=float, default=-60.0)
    return parser.parse_args(argv)


def process_rollouts_dir(rollouts_dir, args):
    """Process one rollouts directory using the original selection behavior."""
    candidates = find_eval_candidates(rollouts_dir, args.pattern)
    best_eval = select_best_eval(candidates)
    print(
        f"Selected {best_eval.eval_dir}: "
        f"{best_eval.success_count}/{best_eval.rollout_count} successful "
        f"({best_eval.success_rate:.2%})"
    )

    processed_rollouts = []
    resumed_rollout_count = 0
    processed_rollout_count = 0
    for npz_path, outcome in tqdm(
        zip(best_eval.rollout_paths, best_eval.outcomes),
        total=best_eval.rollout_count,
        desc=best_eval.eval_dir.name,
    ):
        if args.resume:
            completed = load_completed_rollout(
                npz_path,
                best_eval.eval_dir,
                args,
                outcome,
            )
            if completed is not None:
                processed_rollouts.append(completed)
                resumed_rollout_count += 1
                continue
        processed_rollouts.append(
            process_rollout(
                npz_path,
                best_eval.eval_dir,
                args,
                outcome=outcome,
            )
        )
        processed_rollout_count += 1

    stats_path = save_best_stats(
        rollouts_dir,
        best_eval,
        candidates,
        processed_rollouts,
        args,
    )
    mirrored_stats_path = portable_stats_path(rollouts_dir, args.stats_dir)
    print(f"Saved processed rollout artifacts inside {best_eval.eval_dir}")
    if args.resume:
        print(
            "Resume summary: "
            f"reused {resumed_rollout_count}, "
            f"processed {processed_rollout_count}, "
            f"total {best_eval.rollout_count} rollouts"
        )
    print(f"Saved best evaluation statistics to {stats_path}")
    print(f"Saved portable statistics copy to {mirrored_stats_path}")
    return stats_path


def main():
    args = parse_args()
    rollout_dirs = discover_rollouts_dirs(args.input_dir)
    print(
        f"Discovered {len(rollout_dirs)} rollout director"
        f"{'y' if len(rollout_dirs) == 1 else 'ies'} under "
        f"{args.input_dir.resolve()}"
    )
    for index, rollouts_dir in enumerate(rollout_dirs, start=1):
        print(f"\n[{index}/{len(rollout_dirs)}] Processing {rollouts_dir}")
        process_rollouts_dir(rollouts_dir, args)


if __name__ == "__main__":
    main()
