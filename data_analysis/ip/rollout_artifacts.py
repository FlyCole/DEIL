"""Filesystem validation for paired Instant Policy rollout artifacts."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Tuple


ROLLOUT_STEM_PATTERN = re.compile(r"^rollout_(\d+)_(success|failure)$")


def validate_eval_artifacts(
    eval_dir: Path,
    expected_rollouts: Optional[int] = None,
) -> Tuple[bool, str]:
    """Validate nonempty NPZ/MP4 pairs and their rollout-ID coverage."""
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
