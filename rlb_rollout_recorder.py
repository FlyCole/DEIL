"""DP3-compatible rollout artifact recording for RLBench evaluations.

The recorder intentionally has no RLBench or policy-model dependencies so it
can be tested without launching the simulator or loading a checkpoint.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
import tempfile
from typing import Any, Dict, Iterable, Optional, Set, Tuple

import cv2
import numpy as np


ROLLOUT_SCHEMA_VERSION = 1
ROLLOUT_STEM_PATTERN = re.compile(r"^rollout_(\d+)_(success|failure)$")


def _safe_component(value: Any) -> str:
    """Return a filesystem-safe, human-readable path component."""
    text = str(value).strip()
    safe = "".join(char if char.isalnum() or char in "-_." else "-" for char in text)
    safe = safe.strip("-.")
    return safe or "unknown"


def rollout_config_dir(
    output_root: str,
    task_name: str,
    depth_estimator: str,
    camera_names: Iterable[str],
    resolution: int,
    mask_input: Optional[bool],
) -> Path:
    """Return the config-specific directory that owns ``rollouts/eval_*``."""
    cameras = "+".join(_safe_component(name) for name in camera_names)
    if mask_input is None:
        mask_name = "na"
    else:
        mask_name = "masked" if bool(mask_input) else "unmasked"

    return (
        Path(output_root)
        / _safe_component(task_name)
        / _safe_component(depth_estimator)
        / f"cams-{cameras}"
        / f"res-{int(resolution)}"
        / f"mask-{mask_name}"
    )


def create_eval_dir(config_dir: Path) -> Path:
    """Atomically allocate a DP3-style ``rollouts/eval_NNNN`` directory."""
    rollouts_dir = Path(config_dir) / "rollouts"
    rollouts_dir.mkdir(parents=True, exist_ok=True)

    eval_index = 0
    while True:
        eval_dir = rollouts_dir / f"eval_{eval_index:04d}"
        try:
            eval_dir.mkdir()
            break
        except FileExistsError:
            eval_index += 1

    (eval_dir / "live_states").mkdir()
    (eval_dir / "live_videos").mkdir()
    return eval_dir


def single_eval_dir(config_dir: Path, create: bool = False) -> Path:
    """Return the one evaluation directory owned by a configuration.

    RLBench task evaluation uses exactly one 200-rollout run per configuration,
    so reruns must reuse ``eval_0000`` instead of allocating ``eval_0001`` and
    beyond. ``create`` only ensures the directory structure exists.
    """
    eval_dir = Path(config_dir) / "rollouts" / "eval_0000"
    if create:
        (eval_dir / "live_states").mkdir(parents=True, exist_ok=True)
        (eval_dir / "live_videos").mkdir(parents=True, exist_ok=True)
    return eval_dir


def validate_eval_artifacts(
    eval_dir: Path,
    expected_rollouts: Optional[int] = None,
) -> Tuple[bool, str]:
    """Validate paired, nonempty rollout files and unique rollout IDs."""
    eval_dir = Path(eval_dir)
    state_paths = tuple(sorted((eval_dir / "live_states").glob("rollout_*.npz")))
    video_paths = tuple(sorted((eval_dir / "live_videos").glob("rollout_*.mp4")))
    state_stems = {path.stem for path in state_paths}
    video_stems = {path.stem for path in video_paths}

    if len(state_stems) != len(state_paths) or len(video_stems) != len(video_paths):
        return False, "duplicate rollout artifact stems"
    if state_stems != video_stems:
        return False, "state/video rollout stems do not match"
    if expected_rollouts is not None and len(state_paths) != int(expected_rollouts):
        return False, (
            f"expected {int(expected_rollouts)} rollout pairs, "
            f"found {len(state_paths)}"
        )
    if any(path.stat().st_size == 0 for path in state_paths + video_paths):
        return False, "one or more rollout artifacts are empty"

    rollout_ids = []
    for stem in sorted(state_stems):
        match = ROLLOUT_STEM_PATTERN.fullmatch(stem)
        if match is None:
            return False, f"invalid rollout artifact stem: {stem}"
        rollout_ids.append(int(match.group(1)))
    if len(set(rollout_ids)) != len(rollout_ids):
        return False, "a rollout ID has multiple outcome files"
    if expected_rollouts is not None and set(rollout_ids) != set(
        range(int(expected_rollouts))
    ):
        return False, "rollout IDs are not the expected contiguous range"
    return True, ""


def has_complete_eval(
    config_dir: Path,
    expected_metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    """Whether this config has a complete, matching, intact artifact run."""
    rollouts_dir = Path(config_dir) / "rollouts"
    for manifest_path in sorted(rollouts_dir.glob("eval_*/manifest.json")):
        try:
            with manifest_path.open("r", encoding="utf-8") as file:
                manifest = json.load(file)
        except (OSError, ValueError, TypeError):
            continue
        if manifest.get("schema_version") != ROLLOUT_SCHEMA_VERSION:
            continue
        if manifest.get("status") != "complete":
            continue
        if expected_metadata is not None and any(
            manifest.get(key) != value
            for key, value in expected_metadata.items()
        ):
            continue

        expected_rollouts = manifest.get("recorded_rollouts")
        if not isinstance(expected_rollouts, int) or expected_rollouts <= 0:
            continue
        valid, _ = validate_eval_artifacts(
            manifest_path.parent,
            expected_rollouts=expected_rollouts,
        )
        if valid:
            return True
    return False


def rollout_outcomes(
    eval_dir: Path,
    expected_rollouts: Optional[int] = None,
    allow_orphan_videos: bool = False,
) -> Dict[int, bool]:
    """Return outcomes for intact rollout pairs in a partial or complete run.

    Unlike :func:`validate_eval_artifacts`, this permits fewer than
    ``expected_rollouts`` artifacts so callers can determine which rollout IDs
    remain to be evaluated. Any published artifact inconsistency is rejected;
    silently treating a corrupt pair as resumable could mix incompatible data.
    """
    eval_dir = Path(eval_dir)
    state_paths = tuple(sorted((eval_dir / "live_states").glob("rollout_*.npz")))
    video_paths = tuple(sorted((eval_dir / "live_videos").glob("rollout_*.mp4")))
    state_by_stem = {path.stem: path for path in state_paths}
    video_by_stem = {path.stem: path for path in video_paths}

    if (
        len(state_by_stem) != len(state_paths)
        or len(video_by_stem) != len(video_paths)
    ):
        raise ValueError("duplicate rollout artifact stems")
    state_stems = set(state_by_stem)
    video_stems = set(video_by_stem)
    if state_stems - video_stems:
        raise ValueError("state/video rollout stems do not match")
    if not allow_orphan_videos and video_stems - state_stems:
        raise ValueError("state/video rollout stems do not match")

    outcomes: Dict[int, bool] = {}
    for stem in sorted(state_by_stem):
        state_path = state_by_stem[stem]
        video_path = video_by_stem[stem]
        if state_path.stat().st_size == 0 or video_path.stat().st_size == 0:
            raise ValueError(f"empty rollout artifact pair: {stem}")

        match = ROLLOUT_STEM_PATTERN.fullmatch(stem)
        if match is None:
            raise ValueError(f"invalid rollout artifact stem: {stem}")
        rollout_id = int(match.group(1))
        if (
            expected_rollouts is not None
            and not 0 <= rollout_id < int(expected_rollouts)
        ):
            raise ValueError(
                f"rollout ID {rollout_id} is outside expected range "
                f"0..{int(expected_rollouts) - 1}"
            )
        if rollout_id in outcomes:
            raise ValueError(f"rollout ID {rollout_id} has multiple outcomes")
        outcomes[rollout_id] = match.group(2) == "success"
    return outcomes


def quarantine_orphan_videos(eval_dir: Path) -> Tuple[Path, ...]:
    """Move videos without a published NPZ aside so their IDs can be rerun.

    The recorder intentionally publishes the video before the state file. A
    hard process stop in that small window can therefore leave one final MP4.
    Moving it is recoverable and prevents ``save`` from treating it as a
    completed rollout or refusing to publish the replacement.
    """
    eval_dir = Path(eval_dir)
    state_stems = {
        path.stem for path in (eval_dir / "live_states").glob("rollout_*.npz")
    }
    orphan_videos = tuple(
        sorted(
            path
            for path in (eval_dir / "live_videos").glob("rollout_*.mp4")
            if path.stem not in state_stems
        )
    )
    if not orphan_videos:
        return ()

    quarantine_dir = eval_dir / "incomplete_artifacts"
    quarantine_dir.mkdir(exist_ok=True)
    moved_paths = []
    for source_path in orphan_videos:
        destination = quarantine_dir / source_path.name
        suffix_index = 1
        while destination.exists():
            destination = quarantine_dir / (
                f"{source_path.stem}.resume-{suffix_index}{source_path.suffix}"
            )
            suffix_index += 1
        source_path.rename(destination)
        moved_paths.append(destination)
    return tuple(moved_paths)


def find_resumable_eval(
    config_dir: Path,
    expected_metadata: Dict[str, Any],
    expected_rollouts: int,
    ignored_metadata_keys: Optional[Set[str]] = None,
    allow_manifestless: bool = False,
) -> Optional[Tuple[Path, Dict[int, bool]]]:
    """Find a compatible, intact incomplete evaluation.

    ``recording_code_sha256`` can be explicitly ignored by the caller when a
    resume-only code change should remain compatible with an older partial
    run. All scientific configuration and checkpoint metadata remains strict.

    When ``allow_manifestless`` is true, a fallback scan adopts the newest
    ``eval_*`` directory without a manifest that contains at least one intact
    NPZ/MP4 pair. In that case compatibility is established by ``config_dir``;
    metadata that was never recorded cannot be verified.
    """
    ignored_keys = set(ignored_metadata_keys or ())
    rollouts_dir = Path(config_dir) / "rollouts"
    manifest_paths = sorted(
        rollouts_dir.glob("eval_*/manifest.json"),
        reverse=True,
    )
    for manifest_path in manifest_paths:
        try:
            with manifest_path.open("r", encoding="utf-8") as file:
                manifest = json.load(file)
        except (OSError, ValueError, TypeError):
            continue
        if manifest.get("schema_version") != ROLLOUT_SCHEMA_VERSION:
            continue
        if manifest.get("status") not in {"running", "failed"}:
            continue
        if any(
            manifest.get(key) != value
            for key, value in expected_metadata.items()
            if key not in ignored_keys
        ):
            continue
        try:
            outcomes = rollout_outcomes(
                manifest_path.parent,
                expected_rollouts=expected_rollouts,
                allow_orphan_videos=True,
            )
        except (OSError, ValueError):
            continue
        return manifest_path.parent, outcomes

    if not allow_manifestless:
        return None

    eval_dirs = sorted(
        (
            eval_dir
            for eval_dir in rollouts_dir.glob("eval_*")
            if eval_dir.is_dir() and not (eval_dir / "manifest.json").exists()
        ),
        reverse=True,
    )
    for eval_dir in eval_dirs:
        try:
            outcomes = rollout_outcomes(
                eval_dir,
                expected_rollouts=expected_rollouts,
                allow_orphan_videos=True,
            )
        except (OSError, ValueError):
            continue
        if outcomes:
            return eval_dir, outcomes
    return None


def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
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


def start_manifest(eval_dir: Path, metadata: Dict[str, Any]) -> None:
    payload = dict(metadata)
    payload.update({
        "schema_version": ROLLOUT_SCHEMA_VERSION,
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    _write_json_atomic(Path(eval_dir) / "manifest.json", payload)


def resume_manifest(eval_dir: Path) -> None:
    """Atomically mark a failed or interrupted manifest as running again."""
    manifest_path = Path(eval_dir) / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if payload.get("status") not in {"running", "failed"}:
        raise ValueError(
            f"Cannot resume manifest with status {payload.get('status')!r}"
        )

    previous_error = payload.pop("error", None)
    failed_at_utc = payload.pop("failed_at_utc", None)
    if previous_error is not None:
        failures = list(payload.get("previous_failures", []))
        failures.append({
            "failed_at_utc": failed_at_utc,
            "error": previous_error,
        })
        payload["previous_failures"] = failures
    payload.update({
        "status": "running",
        "resumed_at_utc": datetime.now(timezone.utc).isoformat(),
        "resume_count": int(payload.get("resume_count", 0)) + 1,
    })
    _write_json_atomic(manifest_path, payload)


def complete_manifest(eval_dir: Path, summary: Dict[str, Any]) -> None:
    expected_rollouts = summary.get("recorded_rollouts")
    if not isinstance(expected_rollouts, int) or expected_rollouts <= 0:
        raise ValueError("recorded_rollouts must be a positive integer")
    valid, reason = validate_eval_artifacts(
        eval_dir,
        expected_rollouts=expected_rollouts,
    )
    if not valid:
        raise RuntimeError(f"Cannot complete rollout manifest: {reason}")

    manifest_path = Path(eval_dir) / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    payload.update(summary)
    payload.update({
        "status": "complete",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    _write_json_atomic(manifest_path, payload)


def fail_manifest(eval_dir: Path, error: str) -> None:
    """Mark an allocated evaluation as failed without publishing it as complete."""
    manifest_path = Path(eval_dir) / "manifest.json"
    try:
        with manifest_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
    except (OSError, ValueError, TypeError):
        payload = {"schema_version": ROLLOUT_SCHEMA_VERSION}
    payload.update({
        "status": "failed",
        "failed_at_utc": datetime.now(timezone.utc).isoformat(),
        "error": str(error),
    })
    _write_json_atomic(manifest_path, payload)


class RolloutArtifactRecorder:
    """Buffer and atomically save one DP3-compatible evaluation rollout."""

    def __init__(
        self,
        eval_dir: Path,
        rollout_id: int,
        demo_iter: int,
        rollout_in_demo: int,
        task_name: str,
        depth_estimator: str,
        camera_names: Iterable[str],
        resolution: int,
        mask_input: Optional[bool],
        num_points: int = 2048,
        fps: float = 10.0,
        video_camera: str = "front",
    ):
        if num_points <= 0:
            raise ValueError(f"num_points must be positive, got {num_points}")
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}")

        self.eval_dir = Path(eval_dir)
        self.rollout_id = int(rollout_id)
        self.demo_iter = int(demo_iter)
        self.rollout_in_demo = int(rollout_in_demo)
        self.task_name = str(task_name)
        self.depth_estimator = str(depth_estimator)
        self.camera_names = tuple(str(name) for name in camera_names)
        self.resolution = int(resolution)
        self.mask_input = mask_input
        self.num_points = int(num_points)
        self.fps = float(fps)
        self.video_camera = str(video_camera)

        self._agent_positions = []
        self._transforms = []
        self._policy_point_clouds = []
        self._gt_point_clouds = []
        self._rgb_frames = []
        self._decision_video_frame_indices = []
        self._rgb_frame_shape = None

    @property
    def frame_count(self) -> int:
        return len(self._agent_positions)

    def record_decision(
        self,
        agent_pos: np.ndarray,
        T_w_e: np.ndarray,
        policy_point_cloud: np.ndarray,
        gt_point_cloud: np.ndarray,
        rgb_frame: np.ndarray,
    ) -> None:
        """Record one synchronized policy-decision observation."""
        agent_pos = np.asarray(agent_pos, dtype=np.float32)
        T_w_e = np.asarray(T_w_e, dtype=np.float32)
        policy_point_cloud = np.asarray(policy_point_cloud, dtype=np.float32)
        gt_point_cloud = np.asarray(gt_point_cloud, dtype=np.float32)

        if agent_pos.shape != (7,):
            raise ValueError(f"agent_pos must have shape (7,), got {agent_pos.shape}")
        if T_w_e.shape != (4, 4):
            raise ValueError(f"T_w_e must have shape (4, 4), got {T_w_e.shape}")
        expected_pcd_shape = (self.num_points, 3)
        if policy_point_cloud.shape != expected_pcd_shape:
            raise ValueError(
                "policy_point_cloud must have shape "
                f"{expected_pcd_shape}, got {policy_point_cloud.shape}"
            )
        if gt_point_cloud.shape != expected_pcd_shape:
            raise ValueError(
                f"gt_point_cloud must have shape {expected_pcd_shape}, "
                f"got {gt_point_cloud.shape}"
            )
        named_arrays = (
            ("agent_pos", agent_pos),
            ("T_w_e", T_w_e),
            ("policy_point_cloud", policy_point_cloud),
            ("gt_point_cloud", gt_point_cloud),
        )
        for name, array in named_arrays:
            if not np.isfinite(array).all():
                raise ValueError(f"{name} contains NaN or Inf values")

        rgb_frame = self._validate_rgb_frame(rgb_frame)
        if not self._rgb_frames or not np.array_equal(self._rgb_frames[-1], rgb_frame):
            video_frame_index = self.record_video_frame(rgb_frame)
        else:
            # The previous low-level task.step normally returned the exact
            # observation used by the next policy decision. Reuse that frame
            # instead of duplicating it in the live video.
            video_frame_index = len(self._rgb_frames) - 1

        self._agent_positions.append(agent_pos.copy())
        self._transforms.append(T_w_e.copy())
        self._policy_point_clouds.append(policy_point_cloud.copy())
        self._gt_point_clouds.append(gt_point_cloud.copy())
        self._decision_video_frame_indices.append(video_frame_index)

    def _validate_rgb_frame(self, rgb_frame: np.ndarray) -> np.ndarray:
        rgb_frame = np.asarray(rgb_frame)
        if rgb_frame.ndim != 3 or rgb_frame.shape[2] != 3:
            raise ValueError(f"rgb_frame must have shape (H, W, 3), got {rgb_frame.shape}")
        if rgb_frame.dtype != np.uint8:
            raise ValueError(f"rgb_frame must have dtype uint8, got {rgb_frame.dtype}")
        if self._rgb_frame_shape is not None and rgb_frame.shape != self._rgb_frame_shape:
            raise ValueError(
                f"rgb_frame has shape {rgb_frame.shape}; expected "
                f"{self._rgb_frame_shape}"
            )
        return rgb_frame

    def record_video_frame(self, rgb_frame: np.ndarray) -> int:
        """Append one low-level RGB observation and return its video index."""
        rgb_frame = self._validate_rgb_frame(rgb_frame)
        if self._rgb_frame_shape is None:
            self._rgb_frame_shape = rgb_frame.shape
        self._rgb_frames.append(rgb_frame.copy())
        return len(self._rgb_frames) - 1

    def _payload(self, success: bool) -> Dict[str, np.ndarray]:
        if self.frame_count == 0:
            raise RuntimeError("Cannot save an empty rollout")

        counts = {
            len(self._agent_positions),
            len(self._transforms),
            len(self._policy_point_clouds),
            len(self._gt_point_clouds),
            len(self._decision_video_frame_indices),
        }
        if len(counts) != 1:
            raise RuntimeError("Rollout buffers are not synchronized")
        if not self._rgb_frames:
            raise RuntimeError("Rollout has no RGB video frames")

        video_indices = np.asarray(
            self._decision_video_frame_indices,
            dtype=np.int32,
        )
        if np.any(video_indices < 0) or np.any(video_indices >= len(self._rgb_frames)):
            raise RuntimeError("Decision-to-video frame indices are out of bounds")
        if np.any(np.diff(video_indices) <= 0):
            raise RuntimeError("Decision-to-video frame indices must be strictly increasing")

        return {
            "agent_pos": np.stack(self._agent_positions).astype(np.float32),
            "T_w_e": np.stack(self._transforms).astype(np.float32),
            "obs_pointcloud": np.stack(self._policy_point_clouds).astype(np.float32),
            "gt_pointcloud": np.stack(self._gt_point_clouds).astype(np.float32),
            "fps": np.asarray(self.fps, dtype=np.float32),
            "point_cloud_frame": np.asarray("eef"),
            "pose_frame": np.asarray("world"),
            "agent_pos_format": np.asarray("xyz_rotvec_gripper"),
            "state_sampling": np.asarray("policy_decision"),
            "video_sampling": np.asarray("environment_step"),
            "decision_video_frame_indices": video_indices,
            "video_camera": np.asarray(self.video_camera),
            "task_name": np.asarray(self.task_name),
            "depth_estimator": np.asarray(self.depth_estimator),
            "camera_names": np.asarray(self.camera_names),
            "resolution": np.asarray(self.resolution, dtype=np.int32),
            "mask_input": np.asarray("none" if self.mask_input is None else str(bool(self.mask_input)).lower()),
            "demo_iter": np.asarray(self.demo_iter, dtype=np.int32),
            "rollout_in_demo": np.asarray(self.rollout_in_demo, dtype=np.int32),
            "rollout_id": np.asarray(self.rollout_id, dtype=np.int32),
            "success": np.asarray(bool(success), dtype=np.bool_),
            "decision_step": np.arange(self.frame_count, dtype=np.int32),
        }

    @staticmethod
    def _write_video_atomic(path: Path, frames: Tuple[np.ndarray, ...], fps: float) -> None:
        first_frame = frames[0]
        height, width = first_frame.shape[:2]
        temp_path = None
        writer = None
        try:
            with tempfile.NamedTemporaryFile(
                prefix=f".{path.stem}.",
                suffix=".mp4",
                dir=path.parent,
                delete=False,
            ) as file:
                temp_path = Path(file.name)

            writer = cv2.VideoWriter(
                str(temp_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                float(fps),
                (width, height),
            )
            if not writer.isOpened():
                raise RuntimeError(f"Could not open video writer for {path}")

            for frame_idx, frame in enumerate(frames):
                if frame.dtype != np.uint8 or frame.shape != first_frame.shape:
                    raise ValueError(
                        f"RGB frame {frame_idx} has dtype/shape "
                        f"{frame.dtype}/{frame.shape}; expected "
                        f"uint8/{first_frame.shape}"
                    )
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            writer.release()
            writer = None
            os.replace(temp_path, path)
            temp_path = None
        finally:
            if writer is not None:
                writer.release()
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

    @staticmethod
    def _write_npz_atomic(path: Path, payload: Dict[str, np.ndarray]) -> None:
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                prefix=f".{path.stem}.",
                suffix=".npz",
                dir=path.parent,
                delete=False,
            ) as file:
                temp_path = Path(file.name)
            np.savez_compressed(temp_path, **payload)
            os.replace(temp_path, path)
            temp_path = None
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

    def save(self, success: bool) -> Tuple[Path, Path]:
        """Save paired MP4/NPZ artifacts and return ``(state_path, video_path)``."""
        outcome = "success" if success else "failure"
        stem = f"rollout_{self.rollout_id:04d}_{outcome}"
        state_path = self.eval_dir / "live_states" / f"{stem}.npz"
        video_path = self.eval_dir / "live_videos" / f"{stem}.mp4"
        existing_states = tuple(
            (self.eval_dir / "live_states").glob(
                f"rollout_{self.rollout_id:04d}_*.npz"
            )
        )
        existing_videos = tuple(
            (self.eval_dir / "live_videos").glob(
                f"rollout_{self.rollout_id:04d}_*.mp4"
            )
        )
        if existing_states or existing_videos:
            raise FileExistsError(f"Refusing to overwrite rollout artifact {stem}")

        payload = self._payload(success)
        self._write_video_atomic(video_path, tuple(self._rgb_frames), self.fps)
        try:
            # State is published last so analysis never discovers an NPZ whose
            # paired RGB video has not finished writing.
            self._write_npz_atomic(state_path, payload)
        except Exception:
            video_path.unlink(missing_ok=True)
            raise
        return state_path, video_path
