import json
import zarr
import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm


# ----------------------------
# Chamfer (mean L2, symmetric)
# ----------------------------
def chamfer_kdtree(A, B):
    """
    A, B: non-empty (N, 3) arrays.

    Returns symmetric, unsquared L2 Chamfer distance:
      mean_nn_l2(A->B) + mean_nn_l2(B->A)
    """
    d_ab, d_ba = directional_nearest_neighbor_distances(A, B)
    return float(d_ab.mean() + d_ba.mean())


def directional_nearest_neighbor_distances(A, B):
    """Return pointwise ``A->B`` and ``B->A`` L2 distances."""
    A = np.asarray(A)
    B = np.asarray(B)
    for name, points in (("A", A), ("B", B)):
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"{name} must have shape (N, 3), got {points.shape}")
        if points.shape[0] == 0:
            raise ValueError(f"{name} must contain at least one point")
        if not np.isfinite(points).all():
            raise ValueError(f"{name} contains non-finite coordinates")

    tree_B = cKDTree(B)
    tree_A = cKDTree(A)
    d_ab, _ = tree_B.query(A, k=1, workers=-1)
    d_ba, _ = tree_A.query(B, k=1, workers=-1)
    return (
        np.asarray(d_ab, dtype=np.float64),
        np.asarray(d_ba, dtype=np.float64),
    )


def extract_xyz(arr):
    # arr: (N, D>=3) or (..., N, D)
    arr = np.asarray(arr)
    if arr.ndim < 2 or arr.shape[-1] < 3:
        raise ValueError(f"Point cloud must end in at least 3 coordinates, got {arr.shape}")
    return np.asarray(arr[..., :3], dtype=np.float32)


# ----------------------------
# Normalization utilities
# ----------------------------
def compute_norm_params(points, mode="unit_sphere", robust=False, eps=1e-8):
    """
    points: (N,3)
    returns (center (3,), scale (scalar))
    """
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise ValueError(f"Expected a non-empty point cloud with shape (N, 3), got {points.shape}")
    if not np.isfinite(points).all():
        raise ValueError("Cannot normalize a point cloud containing non-finite coordinates")

    if robust:
        center = np.median(points, axis=0)
    else:
        center = points.mean(axis=0)

    centered = points - center

    if mode == "unit_sphere":
        # scale so max radius = 1
        radii = np.linalg.norm(centered, axis=1)
        scale = np.max(radii)
    elif mode == "unit_cube":
        # scale so max abs coord = 1
        scale = np.max(np.abs(centered))
    elif mode == "std":
        # scale so average std = 1 (isotropic)
        scale = float(np.mean(np.std(centered, axis=0)))
    else:
        raise ValueError(f"Unknown norm mode: {mode}")

    if not np.isfinite(scale) or scale <= eps:
        raise ValueError(f"Cannot normalize a degenerate point cloud with scale={scale}")
    return center.astype(np.float32), float(scale)


def apply_norm(points, center, scale):
    return (points - center) / scale


def normalize_pair(pred, gt, pair_mode="baseline", norm_mode="unit_sphere", robust=False):
    """
    pred, gt: (N,3)
    pair_mode:
      - "baseline": compute (center,scale) from gt, apply to BOTH
      - "independent": compute params separately for pred and gt (shape-only)
    """
    if pair_mode == "baseline":
        c, s = compute_norm_params(gt, mode=norm_mode, robust=robust)
        return apply_norm(pred, c, s), apply_norm(gt, c, s)
    elif pair_mode == "independent":
        c1, s1 = compute_norm_params(pred, mode=norm_mode, robust=robust)
        c2, s2 = compute_norm_params(gt,   mode=norm_mode, robust=robust)
        return apply_norm(pred, c1, s1), apply_norm(gt, c2, s2)
    else:
        raise ValueError(f"Unknown pair_mode: {pair_mode}")


def load_episode_ranges(root, total_frames):
    """Return validated half-open (start, end) frame ranges for every episode."""
    if "meta" not in root or "episode_ends" not in root["meta"]:
        raise RuntimeError(
            "Missing /meta/episode_ends; equal-per-episode aggregation requires "
            "explicit episode boundaries."
        )

    episode_ends = np.asarray(root["meta/episode_ends"][:], dtype=np.int64)
    if episode_ends.ndim != 1 or episode_ends.size == 0:
        raise RuntimeError("/meta/episode_ends must be a non-empty 1D array")
    if np.any(episode_ends <= 0) or np.any(np.diff(episode_ends) <= 0):
        raise RuntimeError("/meta/episode_ends must be strictly increasing and positive")
    if int(episode_ends[-1]) != total_frames:
        raise RuntimeError(
            "/meta/episode_ends does not cover the point-cloud arrays: "
            f"last episode end={int(episode_ends[-1])}, frames={total_frames}"
        )

    episode_starts = np.concatenate(([0], episode_ends[:-1]))
    return [(int(start), int(end)) for start, end in zip(episode_starts, episode_ends)]


def summarize(values):
    """Population summary for a non-empty, finite 1D array."""
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


# ----------------------------
# Main
# ----------------------------
def main(
    zarr_path: str,
    output_json: str = "chamfer_results.json",
    pair_mode: str = "baseline",     # baseline | independent
    norm_mode: str = "unit_sphere",  # unit_sphere | unit_cube | std
    robust_center: bool = False
):
    root = zarr.open(zarr_path, mode="r")
    base_pc = root["data/point_cloud"]

    if base_pc.ndim != 3 or base_pc.shape[1] == 0 or base_pc.shape[2] < 3:
        raise RuntimeError(
            f"/data/point_cloud must have shape (T, N, D>=3), got {base_pc.shape}"
        )
    total_frames = int(base_pc.shape[0])
    if total_frames == 0:
        raise RuntimeError("/data/point_cloud contains no frames")
    episode_ranges = load_episode_ranges(root, total_frames)

    variants = sorted([
        k for k in root["data"].array_keys()
        if k.startswith("point_cloud_") and k != "point_cloud"
    ])

    if len(variants) == 0:
        raise RuntimeError("No datasets named point_cloud_* found under /data")

    # Frame index t must refer to the same observation for every method. Silently
    # truncating to the shortest array can compare incomplete or misaligned data.
    for name in variants:
        var_pc = root[f"data/{name}"]
        if var_pc.ndim != 3 or var_pc.shape[1] == 0 or var_pc.shape[2] < 3:
            raise RuntimeError(
                f"/data/{name} must have shape (T, N, D>=3), got {var_pc.shape}"
            )
        if int(var_pc.shape[0]) != total_frames:
            raise RuntimeError(
                f"Frame-count mismatch for /data/{name}: prediction has "
                f"{var_pc.shape[0]} frames, baseline has {total_frames}. "
                "Finish or repair the prediction array before evaluation."
            )

    results = {
        "zarr_path": zarr_path,
        "baseline": "data/point_cloud",
        "metric": {
            "name": "symmetric_chamfer_distance",
            "pointwise_distance": "euclidean_l2",
            "squared": False,
            "formula": "mean(min_L2(pred->gt)) + mean(min_L2(gt->pred))",
            "units": "dimensionless after per-frame normalization",
        },
        "aggregation": {
            "primary": "equal_weight_per_episode",
            "summary_fields": "statistics across per-episode mean Chamfer distances",
            "episode_count": len(episode_ranges),
            "frame_count": total_frames,
            "episode_lengths": [end - start for start, end in episode_ranges],
        },
        "normalization": {
            "pair_mode": pair_mode,
            "norm_mode": norm_mode,
            "robust_center": robust_center,
            "note": (
                "pair_mode=baseline applies GT-derived (center,scale) to BOTH GT and prediction "
                "(recommended for comparing methods). "
                "pair_mode=independent normalizes each cloud separately (shape-only)."
            )
        },
        "metrics": {}
    }

    for name in variants:
        var_pc = root[f"data/{name}"]

        per_frame = []
        print(
            f"\nComputing normalized Chamfer for {name} "
            f"(frames={total_frames}, episodes={len(episode_ranges)})"
        )

        for t in tqdm(range(total_frames)):
            gt = extract_xyz(base_pc[t])     # (N,3)
            pr = extract_xyz(var_pc[t])      # (N,3)

            try:
                pr_n, gt_n = normalize_pair(
                    pr,
                    gt,
                    pair_mode=pair_mode,
                    norm_mode=norm_mode,
                    robust=robust_center,
                )
                cd = chamfer_kdtree(pr_n, gt_n)
            except ValueError as exc:
                raise RuntimeError(f"Invalid point clouds for {name} at frame {t}: {exc}") from exc
            per_frame.append(cd)

        per_frame = np.asarray(per_frame, dtype=np.float64)
        per_episode = np.asarray(
            [per_frame[start:end].mean() for start, end in episode_ranges],
            dtype=np.float64,
        )
        episode_summary = summarize(per_episode)
        frame_summary = summarize(per_frame)

        results["metrics"][name] = {
            **episode_summary,
            "per_episode": per_episode.tolist(),
            "per_frame_summary": frame_summary,
            "per_frame": per_frame.tolist(),
        }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Saved results to: {output_json}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--zarr_path", required=True)
    p.add_argument("--output_json", default="chamfer_results.json")
    p.add_argument("--pair_mode", default="baseline", choices=["baseline", "independent"])
    p.add_argument("--norm_mode", default="unit_sphere", choices=["unit_sphere", "unit_cube", "std"])
    p.add_argument("--robust_center", action="store_true")
    args = p.parse_args()

    main(
        zarr_path=args.zarr_path,
        output_json=args.output_json,
        pair_mode=args.pair_mode,
        norm_mode=args.norm_mode,
        robust_center=args.robust_center
    )
