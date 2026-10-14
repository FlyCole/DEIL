"""Point-cloud normalization and Chamfer metrics used by IP analysis."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def chamfer_kdtree(first, second):
    """Return symmetric unsquared nearest-neighbor L2 Chamfer distance."""
    first_to_second, second_to_first = directional_nearest_neighbor_distances(
        first,
        second,
    )
    return float(first_to_second.mean() + second_to_first.mean())


def directional_nearest_neighbor_distances(first, second):
    """Return pointwise ``first→second`` and ``second→first`` L2 distances."""
    first = np.asarray(first)
    second = np.asarray(second)
    for name, points in (("first", first), ("second", second)):
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"{name} must have shape (N, 3), got {points.shape}")
        if points.shape[0] == 0:
            raise ValueError(f"{name} must contain at least one point")
        if not np.isfinite(points).all():
            raise ValueError(f"{name} contains non-finite coordinates")

    second_tree = cKDTree(second)
    first_tree = cKDTree(first)
    first_to_second, _ = second_tree.query(first, k=1, workers=-1)
    second_to_first, _ = first_tree.query(second, k=1, workers=-1)
    return (
        np.asarray(first_to_second, dtype=np.float64),
        np.asarray(second_to_first, dtype=np.float64),
    )


def extract_xyz(array):
    """Return the first three coordinates from a point-cloud array."""
    array = np.asarray(array)
    if array.ndim < 2 or array.shape[-1] < 3:
        raise ValueError(
            "Point cloud must end in at least three coordinates, "
            f"got {array.shape}"
        )
    return np.asarray(array[..., :3], dtype=np.float32)


def compute_norm_params(points, mode="unit_sphere", robust=False, eps=1e-8):
    """Return an isotropic normalization center and scale."""
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise ValueError(
            f"Expected a non-empty point cloud with shape (N, 3), got {points.shape}"
        )
    if not np.isfinite(points).all():
        raise ValueError("Cannot normalize non-finite point-cloud coordinates")

    center = np.median(points, axis=0) if robust else points.mean(axis=0)
    centered = points - center
    if mode == "unit_sphere":
        scale = np.max(np.linalg.norm(centered, axis=1))
    elif mode == "unit_cube":
        scale = np.max(np.abs(centered))
    elif mode == "std":
        scale = float(np.mean(np.std(centered, axis=0)))
    else:
        raise ValueError(f"Unknown norm mode: {mode}")
    if not np.isfinite(scale) or scale <= eps:
        raise ValueError(f"Cannot normalize a degenerate cloud with scale={scale}")
    return center.astype(np.float32), float(scale)


def normalize_pair(
    prediction,
    reference,
    pair_mode="baseline",
    norm_mode="unit_sphere",
    robust=False,
):
    """Normalize a predicted/reference point-cloud pair."""
    if pair_mode == "baseline":
        center, scale = compute_norm_params(
            reference,
            mode=norm_mode,
            robust=robust,
        )
        return (
            (prediction - center) / scale,
            (reference - center) / scale,
        )
    if pair_mode == "independent":
        prediction_center, prediction_scale = compute_norm_params(
            prediction,
            mode=norm_mode,
            robust=robust,
        )
        reference_center, reference_scale = compute_norm_params(
            reference,
            mode=norm_mode,
            robust=robust,
        )
        return (
            (prediction - prediction_center) / prediction_scale,
            (reference - reference_center) / reference_scale,
        )
    raise ValueError(f"Unknown pair_mode: {pair_mode}")


def summarize(values):
    """Return population statistics for a non-empty finite vector."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Summary values must be a non-empty, finite 1D array")
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std()),
        "p95": float(np.percentile(values, 95)),
    }
