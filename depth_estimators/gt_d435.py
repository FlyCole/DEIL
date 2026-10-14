"""D435-like disparity corruption for clean metric GT depth.

This is an ablation baseline rather than a complete RealSense simulator.  It
models the stereo depth error that grows quadratically with range by applying
noise, bias, and sub-pixel quantization in disparity space before converting
back to metric depth.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Optional

import numpy as np


D435_BASELINE_M = 0.05
D435_DISPARITY_QUANTIZATION_PX = 1.0 / 32.0
D435_MILD_NOISE_STD_PX = 0.05
D435_MILD_BIAS_PX = 0.02
D435_STRONG_NOISE_STD_PX = 0.15
D435_STRONG_BIAS_PX = 0.08

_GT_D435_PATTERN = re.compile(r"^gt_d435(?:_(?P<preset>mild|strong))?$")


@dataclass(frozen=True)
class D435CorruptionConfig:
    """Parameters for the disparity-space corruption model."""

    baseline_m: float = D435_BASELINE_M
    disparity_noise_std_px: float = D435_MILD_NOISE_STD_PX
    disparity_bias_px: float = D435_MILD_BIAS_PX
    disparity_quantization_px: float = D435_DISPARITY_QUANTIZATION_PX

    def __post_init__(self) -> None:
        values = {
            "baseline_m": self.baseline_m,
            "disparity_noise_std_px": self.disparity_noise_std_px,
            "disparity_bias_px": self.disparity_bias_px,
            "disparity_quantization_px": self.disparity_quantization_px,
        }
        for name, value in values.items():
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value}")
        if self.baseline_m <= 0.0:
            raise ValueError(f"baseline_m must be positive, got {self.baseline_m}")
        if self.disparity_noise_std_px < 0.0:
            raise ValueError(
                "disparity_noise_std_px must be nonnegative, got "
                f"{self.disparity_noise_std_px}"
            )
        if self.disparity_quantization_px <= 0.0:
            raise ValueError(
                "disparity_quantization_px must be positive, got "
                f"{self.disparity_quantization_px}"
            )


def normalize_gt_d435_name(name: str) -> str:
    """Normalize case and hyphenated aliases for stable Zarr keys."""
    return str(name).strip().casefold().replace("-", "_")


def is_gt_d435_name(name: str) -> bool:
    return _GT_D435_PATTERN.fullmatch(normalize_gt_d435_name(name)) is not None


def d435_config_from_name(
    name: str,
    *,
    baseline_m: float = D435_BASELINE_M,
    disparity_noise_std_px: float = D435_MILD_NOISE_STD_PX,
    disparity_bias_px: float = D435_MILD_BIAS_PX,
    disparity_quantization_px: float = D435_DISPARITY_QUANTIZATION_PX,
) -> D435CorruptionConfig:
    """Resolve a named preset, or parameters for the plain ``gt_d435`` mode.

    ``gt_d435_mild`` and ``gt_d435_strong`` fix the noise and bias to the
    reproducible presets below.  Plain ``gt_d435`` uses the supplied noise and
    bias, which default to the mild preset.  Baseline and quantization remain
    configurable for controlled ablations, with requested defaults of 0.05 m
    and 1/32 pixel.
    """
    normalized = normalize_gt_d435_name(name)
    match = _GT_D435_PATTERN.fullmatch(normalized)
    if match is None:
        raise ValueError(
            f"Invalid D435 mode {name!r}; expected gt_d435, "
            "gt_d435_mild, or gt_d435_strong."
        )

    preset = match.group("preset")
    if preset == "mild":
        disparity_noise_std_px = D435_MILD_NOISE_STD_PX
        disparity_bias_px = D435_MILD_BIAS_PX
    elif preset == "strong":
        disparity_noise_std_px = D435_STRONG_NOISE_STD_PX
        disparity_bias_px = D435_STRONG_BIAS_PX

    return D435CorruptionConfig(
        baseline_m=float(baseline_m),
        disparity_noise_std_px=float(disparity_noise_std_px),
        disparity_bias_px=float(disparity_bias_px),
        disparity_quantization_px=float(disparity_quantization_px),
    )


def corrupt_gt_depth_d435(
    depth_m: np.ndarray,
    fx_px: float,
    config: D435CorruptionConfig,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Corrupt metric depth through quantized stereo disparity.

    For each positive finite depth ``z``, disparity is ``d = |fx| * B / z``.
    Gaussian noise and a fixed bias are added in pixels, disparity is rounded
    to the configured sub-pixel step, and depth is recovered as
    ``z' = |fx| * B / d'``. Signed camera-intrinsics conventions are therefore
    supported without changing their projection matrices. Invalid input pixels
    remain zero.
    """
    depth_m = np.asarray(depth_m, dtype=np.float32)
    fx_px = float(fx_px)
    if not np.isfinite(fx_px) or fx_px == 0.0:
        raise ValueError(f"fx_px must be finite and nonzero, got {fx_px}")

    # RLBench may encode an image-axis flip with a negative focal length.
    # Stereo disparity depends on the physical focal-length magnitude.
    fx_px = abs(fx_px)

    output = np.zeros_like(depth_m, dtype=np.float32)
    valid_depth = np.isfinite(depth_m) & (depth_m > 0.0)
    if not np.any(valid_depth):
        return output

    scale = fx_px * config.baseline_m
    disparity = scale / depth_m[valid_depth].astype(np.float64)
    if config.disparity_noise_std_px > 0.0:
        generator = rng if rng is not None else np.random.default_rng()
        disparity += generator.normal(
            0.0,
            config.disparity_noise_std_px,
            size=disparity.shape,
        )
    disparity += config.disparity_bias_px
    disparity = (
        np.rint(disparity / config.disparity_quantization_px)
        * config.disparity_quantization_px
    )

    valid_disparity = np.isfinite(disparity) & (disparity > 0.0)
    corrupted = np.zeros_like(disparity, dtype=np.float32)
    corrupted[valid_disparity] = (
        scale / disparity[valid_disparity]
    ).astype(np.float32)
    output[valid_depth] = corrupted
    return output


class GTD435Depth:
    """GT-pipeline marker that applies D435-like disparity corruption."""

    uses_gt_depth = True

    def __init__(
        self,
        name: str = "gt_d435",
        *,
        baseline_m: float = D435_BASELINE_M,
        disparity_noise_std_px: float = D435_MILD_NOISE_STD_PX,
        disparity_bias_px: float = D435_MILD_BIAS_PX,
        disparity_quantization_px: float = D435_DISPARITY_QUANTIZATION_PX,
        seed: int = 0,
    ):
        self.name = normalize_gt_d435_name(name)
        self.config = d435_config_from_name(
            self.name,
            baseline_m=baseline_m,
            disparity_noise_std_px=disparity_noise_std_px,
            disparity_bias_px=disparity_bias_px,
            disparity_quantization_px=disparity_quantization_px,
        )
        self.base_seed = int(seed)
        self._rng = np.random.default_rng(self.base_seed)

    def reset(self, seed: Optional[int] = None) -> None:
        noise_seed = (
            self.base_seed
            if seed is None
            else np.random.SeedSequence([self.base_seed, int(seed)])
        )
        self._rng = np.random.default_rng(noise_seed)

    def corrupt_depth(
        self,
        depth_m: np.ndarray,
        intrinsics: np.ndarray,
    ) -> np.ndarray:
        intrinsics = np.asarray(intrinsics)
        if intrinsics.shape != (3, 3):
            raise ValueError(
                f"intrinsics must have shape (3, 3), got {intrinsics.shape}"
            )
        return corrupt_gt_depth_d435(
            depth_m,
            float(intrinsics[0, 0]),
            self.config,
            self._rng,
        )
