"""Deterministic additive noise for GT-depth ablation experiments."""

from __future__ import annotations

import re
from typing import Optional

import numpy as np


DEFAULT_GT_NOISE_STD_M = 0.010
_GT_NOISY_PATTERN = re.compile(
    r"^gt_noisy(?:_(?P<std_mm>\d+(?:p\d+)?)mm)?$"
)


def normalize_gt_noisy_name(name: str) -> str:
    """Normalize hyphenated aliases without changing the encoded noise level."""
    return str(name).strip().casefold().replace("-", "_")


def is_gt_noisy_name(name: str) -> bool:
    return _GT_NOISY_PATTERN.fullmatch(normalize_gt_noisy_name(name)) is not None


def gt_noise_std_m_from_name(
    name: str,
    default_std_m: float = DEFAULT_GT_NOISE_STD_M,
) -> float:
    """Resolve ``gt_noisy[_<N>mm]`` to a Gaussian standard deviation."""
    normalized = normalize_gt_noisy_name(name)
    match = _GT_NOISY_PATTERN.fullmatch(normalized)
    if match is None:
        raise ValueError(
            f"Invalid GT-noisy mode {name!r}; expected gt_noisy or "
            "gt_noisy_<N>mm (for example gt_noisy_5mm)."
        )
    encoded_mm = match.group("std_mm")
    std_m = (
        float(encoded_mm.replace("p", ".")) / 1000.0
        if encoded_mm is not None else float(default_std_m)
    )
    if not np.isfinite(std_m) or std_m < 0.0:
        raise ValueError(f"GT depth noise standard deviation must be >= 0, got {std_m}")
    return std_m


def add_gt_depth_noise(
    depth: np.ndarray,
    std_m: float,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Add Gaussian noise to valid metric-depth pixels and preserve invalids.

    Positive finite pixels are considered valid. Invalid pixels are returned as
    zero, and noisy values are clipped at zero before point-cloud reprojection.
    """
    depth = np.asarray(depth, dtype=np.float32)
    if not np.isfinite(std_m) or std_m < 0.0:
        raise ValueError(f"std_m must be a finite nonnegative value, got {std_m}")
    output = np.zeros_like(depth, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0.0)
    output[valid] = depth[valid]
    if std_m > 0.0 and np.any(valid):
        generator = rng if rng is not None else np.random.default_rng()
        noise = generator.normal(0.0, float(std_m), size=int(valid.sum()))
        output[valid] += noise.astype(np.float32)
        np.maximum(output, 0.0, out=output)
    return output


class GTNoisyDepth:
    """GT-pipeline marker that perturbs depth immediately before reprojection."""

    uses_gt_depth = True

    def __init__(
        self,
        name: str = "gt_noisy",
        default_std_m: float = DEFAULT_GT_NOISE_STD_M,
        seed: int = 0,
    ):
        self.name = normalize_gt_noisy_name(name)
        self.std_m = gt_noise_std_m_from_name(self.name, default_std_m)
        self.base_seed = int(seed)
        self._rng = np.random.default_rng(self.base_seed)

    def reset(self, seed: Optional[int] = None) -> None:
        noise_seed = (
            self.base_seed
            if seed is None
            else np.random.SeedSequence([self.base_seed, int(seed)])
        )
        self._rng = np.random.default_rng(noise_seed)

    def add_noise(self, depth: np.ndarray) -> np.ndarray:
        return add_gt_depth_noise(depth, self.std_m, self._rng)

    def corrupt_depth(
        self,
        depth: np.ndarray,
        intrinsics: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Apply corruption through the shared online GT-depth interface."""
        del intrinsics
        return self.add_noise(depth)
