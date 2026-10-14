import argparse
from contextlib import nullcontext
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from dp3_derived_transaction import DerivedPCDTransaction, recover_pending_publication
import numpy as np
import zarr
import cv2
import open3d as o3d
from tqdm import tqdm
from termcolor import cprint
import time

from rlb_cameras import RLBCameras
from pcd_scaler import scale_icp
from utils import visualise_pcds, pose_6dof_to_transform
from utils import transform_pcd
from depth_estimators.gt_d435 import (
    D435_BASELINE_M,
    D435_DISPARITY_QUANTIZATION_PX,
    D435_MILD_BIAS_PX,
    D435_MILD_NOISE_STD_PX,
    corrupt_gt_depth_d435,
    d435_config_from_name,
    is_gt_d435_name,
    normalize_gt_d435_name,
)
from depth_estimators.gt_noisy import (
    DEFAULT_GT_NOISE_STD_M,
    add_gt_depth_noise,
    gt_noise_std_m_from_name,
    is_gt_noisy_name,
    normalize_gt_noisy_name,
)

# PyTorch3D supplies farthest-point sampling on CPU or CUDA.
import torch
import pytorch3d.ops as torch3d_ops

ESTIMATOR_ALIASES = {
    "mapanything": "map_anything",
    "map-anything": "map_anything",
}


# --------------------------------------------------------------------------------------
# FPS to fixed N
# --------------------------------------------------------------------------------------
def farthest_point_sampling(points, num_points=1024, use_cuda=True):
    points = np.asarray(points, dtype=np.float32)
    K = [num_points]
    if use_cuda:
        points = torch.from_numpy(points).cuda()
        sampled_points, indices = torch3d_ops.sample_farthest_points(points=points.unsqueeze(0), K=K)
        sampled_points = sampled_points.squeeze(0)
        sampled_points = sampled_points.cpu().numpy()
    else:
        points = torch.from_numpy(points)
        sampled_points, indices = torch3d_ops.sample_farthest_points(points=points.unsqueeze(0), K=K)
        sampled_points = sampled_points.squeeze(0)
        sampled_points = sampled_points.numpy()

    return sampled_points, indices

def preprocess_point_cloud(points, num_points=1024, use_cuda=True, return_color=False):
    points = np.asarray(points, dtype=np.float32)
    points = points.reshape(-1, points.shape[-1])
    points = points[np.isfinite(points[:, :3]).all(axis=1)]
    has_color = points.shape[1] >= 6
    points_xyz = points[:, :3]

    if points_xyz.shape[0] == 0:
        fixed_xyz = np.zeros((num_points, 3), dtype=np.float32)
        if return_color:
            return np.hstack((fixed_xyz, np.zeros((num_points, 3), dtype=np.float32)))
        return fixed_xyz

    if points_xyz.shape[0] < num_points:
        indices = np.random.choice(points_xyz.shape[0], num_points, replace=True)
        fixed_xyz = points_xyz[indices].astype(np.float32)
        if return_color and has_color:
            return np.hstack((fixed_xyz, points[indices, 3:6].astype(np.float32)))
        return fixed_xyz

    fixed_xyz, sample_indices = farthest_point_sampling(points_xyz, num_points, use_cuda)
    fixed_xyz = fixed_xyz.astype(np.float32)
    if return_color and has_color:
        sample_indices = sample_indices.squeeze(0).cpu().numpy()
        return np.hstack((fixed_xyz, points[sample_indices, 3:6].astype(np.float32)))
    return fixed_xyz


def reproject_to_colored_pcd(depth, rgb, intrinsics, extrinsics, mask):
    point_mask = mask
    if point_mask is not None:
        point_mask = point_mask.astype(bool) & (depth > 0)
    else:
        point_mask = depth > 0

    xyz = RLBCameras.reproject_to_pcd(depth, intrinsics, extrinsics, point_mask)
    colors = rgb[point_mask].reshape(-1, 3).astype(np.float32) / 255.0
    finite = np.isfinite(xyz).all(axis=1)
    xyz = xyz[finite]
    colors = colors[finite]
    return np.hstack((xyz.astype(np.float32), colors.astype(np.float32)))


def normalize_estimator_name(name: str) -> str:
    name = name.lower()
    name = ESTIMATOR_ALIASES.get(name, name)
    if name.startswith(("gt_d435", "gt-d435")):
        normalized = normalize_gt_d435_name(name)
        d435_config_from_name(normalized)
        return normalized
    if name.startswith(("gt_noisy", "gt-noisy")):
        normalized = normalize_gt_noisy_name(name)
        gt_noise_std_m_from_name(normalized)
        return normalized
    return name


def make_depth_estimator(name: str):
    if name == 'gt' or is_gt_noisy_name(name) or is_gt_d435_name(name):
        return None
    if name == 'vggt':
        from depth_estimators.vggt_est import VGGT
        return VGGT()
    if name == 'map_anything':
        from depth_estimators.map_anything_est import MapAnythingEst
        return MapAnythingEst()
    if name == 'mast3r':
        from depth_estimators.mast3r_est import Mast3rEst
        return Mast3rEst()
    if name == 'moge':
        from depth_estimators.moge_depth import MoGeDepth
        return MoGeDepth()
    if name == 'pi3':
        from depth_estimators.pi3_est import Pi3Est
        return Pi3Est()
    raise ValueError(f"Unknown estimator: {name}")


def pcd_stats(name: str, points: np.ndarray, warn_abs: float = 10.0):
    points = np.asarray(points)
    if points.size == 0:
        cprint(f"[DEBUG] {name}: empty", "yellow")
        return
    points = points.reshape(-1, points.shape[-1])[:, :3]
    finite = np.isfinite(points).all(axis=1)
    if not finite.any():
        cprint(f"[WARN] {name}: no finite points", "yellow")
        return
    points = points[finite]
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    max_abs = np.abs(points).max()
    msg = (
        f"[DEBUG] {name}: n={points.shape[0]} "
        f"min={np.round(mins, 4)} max={np.round(maxs, 4)} max_abs={max_abs:.4f}"
    )
    cprint(msg, "yellow" if max_abs > warn_abs else "cyan")

# --------------------------------------------------------------------------------------
# Estimators (same interface)
# --------------------------------------------------------------------------------------
def run_estimator(name, imgs, intrinsics, extrinsics):
    name = normalize_estimator_name(name)
    print(f'Running depth estimator: {name}')
    if name == 'gt' or is_gt_noisy_name(name) or is_gt_d435_name(name):
        raise ValueError(
            f"run_estimator({name!r}) should not be called. Use stored GT depth."
        )
    est = make_depth_estimator(name)
    return est.infer(imgs, intrinsics, extrinsics)


# --------------------------------------------------------------------------------------
# Mask erosion removes silhouette pixels before depth reprojection.
# --------------------------------------------------------------------------------------
def erode_masks(masks_bool: np.ndarray, resolution: int) -> np.ndarray:
    out = []
    for c in range(masks_bool.shape[0]):
        if resolution == 512:
            kernel = np.ones((5, 5), np.uint8)
        elif resolution == 256:
            kernel = np.ones((3, 3), np.uint8)
        else:
            kernel = np.ones((2, 2), np.uint8)
        m = cv2.erode(masks_bool[c].astype(np.uint8), kernel, iterations=1).astype(bool)
        out.append(m)
    return np.stack(out, axis=0)


# --------------------------------------------------------------------------------------
# Reconstruct and return FIXED (num_points,3)
# --------------------------------------------------------------------------------------
def reconstruct_fixed_pcd_for_estimator(
    est_name: str,
    imgs: np.ndarray,         # (Ncam,H,W,3) uint8
    depths_gt: np.ndarray,    # (Ncam,H,W) float
    depth_estimator,          # estimator object with .infer() method
    masks_u8: np.ndarray,     # (Ncam,H,W) uint8/bool
    intrinsics: np.ndarray,   # (Ncam,3,3) float
    extrinsics: np.ndarray,   # (Ncam,4,4) float
    resolution: int,
    mask_input: bool,
    num_points: int,
    use_cuda_fps: bool,
    debug: bool,
    debug_name: str = "",
    do_scale_icp: bool = True,
    return_color: bool = False,
    gt_noise_std_m: float = DEFAULT_GT_NOISE_STD_M,
    noise_rng: np.random.Generator = None,
    d435_baseline_m: float = D435_BASELINE_M,
    d435_disparity_noise_std_px: float = D435_MILD_NOISE_STD_PX,
    d435_disparity_bias_px: float = D435_MILD_BIAS_PX,
    d435_disparity_quantization_px: float = D435_DISPARITY_QUANTIZATION_PX,
):
    Ncam = imgs.shape[0]
    masks_bool = erode_masks(masks_u8.astype(bool), resolution)

    # Depth source
    if est_name == "gt":
        depths = depths_gt
    elif is_gt_noisy_name(est_name):
        noise_std_m = gt_noise_std_m_from_name(est_name, gt_noise_std_m)
        depths = add_gt_depth_noise(depths_gt, noise_std_m, noise_rng)
    elif is_gt_d435_name(est_name):
        d435_config = d435_config_from_name(
            est_name,
            baseline_m=d435_baseline_m,
            disparity_noise_std_px=d435_disparity_noise_std_px,
            disparity_bias_px=d435_disparity_bias_px,
            disparity_quantization_px=d435_disparity_quantization_px,
        )
        d435_rng = noise_rng if noise_rng is not None else np.random.default_rng()
        depths = np.stack([
            corrupt_gt_depth_d435(
                depths_gt[c],
                float(intrinsics[c, 0, 0]),
                d435_config,
                d435_rng,
            )
            for c in range(Ncam)
        ], axis=0)
    else:
        imgs_list = [imgs[c] for c in range(Ncam)]
        intr_list = [intrinsics[c] for c in range(Ncam)]
        extr_list = [extrinsics[c] for c in range(Ncam)]

        if mask_input:
            masked_imgs = []
            for c in range(Ncam):
                mi = imgs_list[c].copy()
                mi[~masks_bool[c]] = 0
                masked_imgs.append(mi)
            est_data = depth_estimator.infer(masked_imgs, intr_list, extr_list)
        else:
            est_data = depth_estimator.infer(imgs_list, intr_list, extr_list)


        if "depth" in est_data:
            depths = np.stack([d for d in est_data["depth"]], axis=0)
            if depths.shape[0] != Ncam:
                depths = depths[:Ncam]
        elif "pts3d" in est_data:
            raise RuntimeError(
                f"{est_name} returned 'pts3d' but not 'depth'. "
                f"Adapt here if you want to use pts3d directly."
            )
        else:
            raise RuntimeError(f"{est_name} output missing 'depth'/'pts3d': keys={list(est_data.keys())}")

    # Depth -> per-cam point clouds
    reprojected_pcds = []
    for c in range(Ncam):
        depth = np.asarray(depths[c], dtype=np.float32).copy()
        depth[~np.isfinite(depth)] = 0.0
        depth[depth < 0] = 0.0
        pcd = reproject_to_colored_pcd(depth, imgs[c], intrinsics[c], extrinsics[c], masks_bool[c])
        reprojected_pcds.append(pcd)

    # Outlier removal
    if est_name == "map_anything":
        cleaned = []
        for pcd in reprojected_pcds:
            if pcd.shape[0] < 4:
                cleaned.append(pcd)
                continue
            pcd_o3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd[:, :3]))
            nb_neighbors = int(70 * resolution ** 2 / 700 ** 2)
            nb_neighbors = min(max(nb_neighbors, 3), max(pcd.shape[0] - 1, 1))
            _, ind_sor = pcd_o3d.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=2.0)
            cleaned.append(pcd[np.asarray(ind_sor)])
        reprojected_pcds = cleaned

    # Optional scale alignment (multi-cam learned depth)
    uses_gt_pipeline = (
        est_name == "gt"
        or is_gt_noisy_name(est_name)
        or is_gt_d435_name(est_name)
    )
    if do_scale_icp and not uses_gt_pipeline and len(reprojected_pcds) > 1:
        try:
            H, W = depths.shape[1], depths.shape[2]
            scaled_pcds, _ = scale_icp(
                reprojected_pcds,
                list(extrinsics),
                previous_scale=None,
                n_initialisations=8,
                max_iterations=30,
                intrinsics=((W, H), intrinsics[0]),
                visualise=False,
                verbose=False
            )
            reprojected_pcds = [p for p in scaled_pcds if p.shape[0] != 0]
        except Exception as e:
            print(f"[WARN] scale_icp failed for {est_name}: {e} (using unscaled pcds)")

    if debug:
        gt_pcds = [
            reproject_to_colored_pcd(
                depths_gt[c].copy(),
                imgs[c],
                intrinsics[c],
                extrinsics[c],
                masks_bool[c],
            )
            for c in range(Ncam)
        ]
        for c, pcd in enumerate(reprojected_pcds):
            pcd_stats(f"{debug_name} dense cam{c}", pcd)
        visualise_pcds(
            reprojected_pcds,
            extrinsics,
            intrinsics,
            gt_pcd=[pcd[:, :3] for pcd in gt_pcds],
            voxel_size=0.01,
            window_name=f"{debug_name} dense reconstruction",
        )

    merged = np.concatenate([p for p in reprojected_pcds if p.shape[0] != 0], axis=0) \
        if any(p.shape[0] != 0 for p in reprojected_pcds) else np.zeros((0, 3), dtype=np.float32)

    # Fixed-size sampling
    fixed_xyz = preprocess_point_cloud(
        merged,
        num_points=num_points,
        use_cuda=use_cuda_fps,
        return_color=return_color,
    )
    return fixed_xyz


# --------------------------------------------------------------------------------------
# Planning, input validation, and transactional publication.
def build_parser():
    parser = argparse.ArgumentParser(description='Transactionally rebuild derived DP3 point-cloud arrays.')
    parser.add_argument('--zarr_path', required=True)
    parser.add_argument('--estimators', nargs='+', required=True,
                        help='Derived sources: gt_noisy[_Nmm], gt_d435[_mild|_strong], vggt, pi3, moge, mast3r, map_anything.')
    parser.add_argument('--mask_input', action='store_true')
    parser.add_argument('--resolution', type=int, default=256)
    parser.add_argument('--num_points', type=int, default=1024)
    parser.add_argument('--limit_steps', type=int, default=-1,
                        help='Process a positive prefix length, or -1 for every source step.')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--debug_reprojection', action='store_true')
    parser.add_argument('--debug_every', type=int, default=1)
    parser.add_argument('--no_scale_icp', action='store_true')
    recovery = parser.add_mutually_exclusive_group()
    recovery.add_argument('--resume', action='store_true',
                          help='Resume verified committed rows; legacy unmarked arrays are rejected.')
    recovery.add_argument('--rebuild', action='store_true',
                          help='Rebuild safely in staging (the default); replace outputs only after success.')
    parser.add_argument('--fps_cpu', action='store_true')
    parser.add_argument('--dry_run', action='store_true',
                        help='Validate and run reconstruction with read-only inputs; do not create or write keys.')
    parser.add_argument('--gt_noise_std_m', type=float, default=DEFAULT_GT_NOISE_STD_M)
    parser.add_argument('--gt_noise_seed', type=int, default=0)
    parser.add_argument('--d435_baseline_m', type=float, default=D435_BASELINE_M)
    parser.add_argument('--d435_disparity_noise_std_px', type=float, default=D435_MILD_NOISE_STD_PX)
    parser.add_argument('--d435_disparity_bias_px', type=float, default=D435_MILD_BIAS_PX)
    parser.add_argument('--d435_disparity_quantization_px', type=float, default=D435_DISPARITY_QUANTIZATION_PX)
    parser.add_argument('--d435_noise_seed', type=int, default=0)
    return parser


def validate_arguments(args):
    args.estimators = [normalize_estimator_name(name.strip()) for name in args.estimators]
    learned = {'vggt', 'pi3', 'moge', 'mast3r', 'map_anything'}
    for name in args.estimators:
        if name == 'gt':
            raise ValueError('Use existing /data/point_cloud for GT; plain gt is not a derived output.')
        if name not in learned and not is_gt_noisy_name(name) and not is_gt_d435_name(name):
            raise ValueError(f'Unknown derived estimator: {name}')
    if len(args.estimators) != len(set(args.estimators)):
        raise ValueError('--estimators contains duplicate names or aliases')
    for name in ('resolution', 'num_points', 'debug_every'):
        if getattr(args, name) <= 0:
            raise ValueError(f'--{name} must be positive')
    if args.limit_steps != -1 and args.limit_steps <= 0:
        raise ValueError('--limit_steps must be positive or -1')
    if not np.isfinite(args.gt_noise_std_m) or args.gt_noise_std_m < 0:
        raise ValueError('--gt_noise_std_m must be finite and nonnegative')
    for name in ('gt_noise_seed', 'd435_noise_seed'):
        if not 0 <= getattr(args, name) < 2**32:
            raise ValueError(f'--{name} must be between 0 and 4294967295')
    d435_config_from_name('gt_d435', baseline_m=args.d435_baseline_m,
        disparity_noise_std_px=args.d435_disparity_noise_std_px,
        disparity_bias_px=args.d435_disparity_bias_px,
        disparity_quantization_px=args.d435_disparity_quantization_px)
    return args


def validate_input_schema(root, resolution, limit_steps):
    if 'data' not in root or 'meta' not in root:
        raise RuntimeError('Expected /data and /meta groups in the existing Zarr dataset.')
    data, meta = root['data'], root['meta']
    if not isinstance(data, zarr.Group) or not isinstance(meta, zarr.Group):
        raise RuntimeError('/data and /meta must be Zarr groups.')
    required = ('img', 'depth', 'mask', 'cam_extrinsics', 'state')
    for name in required:
        if name not in data or not isinstance(data[name], zarr.Array):
            raise RuntimeError(f'Missing source array /data/{name}.')
    if 'cam_intrinsics' not in meta or not isinstance(meta['cam_intrinsics'], zarr.Array):
        raise RuntimeError('Missing /meta/cam_intrinsics; recollect with camera calibration.')
    image = data['img']
    if image.ndim != 5 or image.shape[-1] != 3 or image.dtype != np.dtype('uint8'):
        raise RuntimeError('/data/img must be uint8 with shape (T,Ncam,H,W,3).')
    steps, cameras, height, width, _ = image.shape
    if min(steps, cameras, height, width) <= 0:
        raise RuntimeError('Source observations must have positive steps, cameras, and image size.')
    if (height, width) != (resolution, resolution):
        raise RuntimeError(f'--resolution {resolution} does not match stored images {height}x{width}.')
    expected = {'depth': (steps, cameras, height, width),
                'mask': (steps, cameras, height, width),
                'cam_extrinsics': (steps, cameras, 4, 4), 'state': (steps, 7)}
    for name, shape in expected.items():
        if data[name].shape != shape:
            raise RuntimeError(f'/data/{name} shape {data[name].shape} does not match {shape}.')
        if data[name].dtype.kind not in ('bui' if name == 'mask' else 'fi'):
            raise RuntimeError(f'/data/{name} has a nonnumeric or unsupported dtype.')
    intrinsics = meta['cam_intrinsics'][:]
    if (intrinsics.shape != (cameras, 3, 3) or intrinsics.dtype.kind not in 'fi'
            or not np.isfinite(intrinsics).all()):
        raise RuntimeError('/meta/cam_intrinsics must contain finite (Ncam,3,3) matrices.')
    if np.any(intrinsics[:, 0, 0] == 0) or np.any(intrinsics[:, 1, 1] == 0):
        raise RuntimeError('Camera focal lengths must be nonzero.')
    target_steps = steps if limit_steps < 0 else min(steps, limit_steps)
    for start in range(0, target_steps, 100):
        stop = min(start + 100, target_steps)
        for name in ('state', 'cam_extrinsics'):
            if not np.isfinite(data[name][start:stop]).all():
                raise RuntimeError(f'/data/{name} contains nonfinite values in steps {start}:{stop}.')
    source_paths = [f'data/{name}' for name in required] + ['meta/cam_intrinsics']
    if 'episode_ends' in meta:
        ends = meta['episode_ends'][:]
        if (ends.ndim != 1 or ends.dtype.kind not in 'iu' or len(ends) == 0
                or ends[0] <= 0 or np.any(ends[1:] <= ends[:-1]) or ends[-1] != steps):
            raise RuntimeError('/meta/episode_ends does not commit every source observation; resume the collector first.')
        source_paths.append('meta/episode_ends')
    schema = {path: {'shape': list(root[path].shape), 'dtype': str(root[path].dtype),
                     'chunks': list(root[path].chunks), 'attrs': dict(root[path].attrs)}
              for path in source_paths}
    fingerprint = hashlib.sha256(json.dumps(schema, sort_keys=True, allow_nan=False).encode())
    fingerprint.update(np.ascontiguousarray(intrinsics).tobytes())
    # Read file stats only under explicit source-array directories. Replacing or
    # resizing source chunks invalidates a resume without rereading all RGB data.
    for source_path in sorted(source_paths):
        directory = Path(root.store.path) / source_path
        for path in sorted(directory.rglob('*')):
            if path.is_file():
                stat = path.stat()
                fingerprint.update(json.dumps([path.relative_to(directory).as_posix(),
                    stat.st_size, stat.st_mtime_ns], separators=(',', ':')).encode())
    return data, intrinsics, target_steps, {
        'schema': schema, 'fingerprint_sha256': fingerprint.hexdigest(),
        'source_steps': steps, 'num_cameras': cameras,
    }


def processing_code_digest(estimator):
    paths = ['rlb_demo_process_dp3.py', 'dp3_derived_transaction.py',
             'rlb_cameras.py', 'pcd_scaler.py', 'utils.py',
             'depth_estimators/base_estimator.py', 'depth_estimators/dependency_paths.py',
             'depth_dependencies.json',
             'depth_estimators/gt_noisy.py', 'depth_estimators/gt_d435.py']
    adapters = {'vggt': 'vggt_est.py', 'pi3': 'pi3_est.py', 'moge': 'moge_depth.py',
                'mast3r': 'mast3r_est.py', 'map_anything': 'map_anything_est.py'}
    if estimator in adapters:
        paths.append('depth_estimators/' + adapters[estimator])
    digest = hashlib.sha256()
    for name in sorted(paths):
        digest.update(name.encode())
        digest.update((Path(__file__).resolve().parent / name).read_bytes())
    return digest.hexdigest()


def loaded_model_identity(estimator, adapter):
    """Hash the actual loaded weights once without modifying model tensors.

    Repository/model names cannot detect a replaced checkpoint. State names,
    dtypes, shapes and all tensor bytes identify parameters and persistent
    buffers. Hash one CPU-contiguous tensor at a time in bounded byte slices;
    no model serialization or additional complete state-dict copy is needed.
    """
    if is_gt_noisy_name(estimator) or is_gt_d435_name(estimator):
        return None
    model = getattr(adapter, 'model', None)
    if not callable(getattr(model, 'state_dict', None)):
        raise ValueError(f'{estimator}: cannot verify actual loaded model weights; expected adapter.model.state_dict().')
    state = model.state_dict()
    if not state:
        raise ValueError(f'{estimator}: the loaded model has an empty state_dict; actual weights cannot be verified.')
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name]
        if (not isinstance(name, str) or not isinstance(tensor, torch.Tensor)
                or tensor.layout != torch.strided or tensor.is_quantized):
            raise ValueError(f'{estimator}: unsupported state_dict entry {name!r}; expected a dense tensor.')
        header = {'name': name, 'dtype': str(tensor.dtype), 'shape': list(tensor.shape)}
        digest.update(json.dumps(header, sort_keys=True, separators=(',', ':')).encode())
        digest.update(b'\0')
        byte_tensor = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        view = memoryview(byte_tensor.numpy())
        for offset in range(0, len(view), 8 * 1024 * 1024):
            digest.update(view[offset:offset + 8 * 1024 * 1024])
        digest.update(b'\0')
        del view, byte_tensor
    return {'model_class': f'{type(model).__module__}.{type(model).__qualname__}',
            'state_dict_sha256': digest.hexdigest(), 'state_tensor_count': len(state)}


def processing_config(args, estimator, source, use_cuda_fps, model_identity=None):
    noise = {}
    if is_gt_noisy_name(estimator):
        noise = dict(depth_source='gt', depth_noise_type='additive_gaussian',
                     gt_noise_std_m=float(gt_noise_std_m_from_name(estimator, args.gt_noise_std_m)),
                     gt_noise_seed=int(args.gt_noise_seed))
    elif is_gt_d435_name(estimator):
        config = d435_config_from_name(estimator, baseline_m=args.d435_baseline_m,
            disparity_noise_std_px=args.d435_disparity_noise_std_px,
            disparity_bias_px=args.d435_disparity_bias_px,
            disparity_quantization_px=args.d435_disparity_quantization_px)
        noise = dict(depth_source='gt', depth_noise_type='d435_disparity_v1',
            d435_baseline_m=float(config.baseline_m),
            d435_disparity_noise_std_px=float(config.disparity_noise_std_px),
            d435_disparity_bias_px=float(config.disparity_bias_px),
            d435_disparity_quantization_px=float(config.disparity_quantization_px),
            d435_noise_seed=int(args.d435_noise_seed))
    return dict(estimator=estimator, source=source, resolution=int(args.resolution),
                num_points=int(args.num_points), mask_input=bool(args.mask_input),
                scale_icp=not args.no_scale_icp, fps_device='cuda' if use_cuda_fps else 'cpu',
                point_cloud_frame='eef', noise_metadata=noise,
                model_identity=model_identity,
                processing_code_sha256=processing_code_digest(estimator))


def preflight_static_configuration(root, configs, num_points, steps, resume):
    """Reject incompatible source/settings or unmarked legacy rows before loading models.

    At this stage previous model identities are only placeholders. The actual
    newly loaded identity is compared by the complete preflight below.
    """
    provisional = deepcopy(configs)
    probe = DerivedPCDTransaction(root, provisional, num_points, steps, resume)
    pending = probe._read_state()
    if resume:
        previous = pending['configs'] if pending else {
            key: root['data'][key].attrs.get('derived_processing_config', {})
            for key in configs if key in root['data']
        }
        for key, config in provisional.items():
            prior = previous.get(key, {})
            if isinstance(prior, dict):
                config['model_identity'] = prior.get('model_identity')
    return DerivedPCDTransaction(root, provisional, num_points, steps, resume).preflight()


def reconstruct_step(args, estimator, depth_estimator, step, imgs, depths, masks,
                     intrinsics, extrinsics, states, use_cuda_fps):
    debug = args.debug and step % args.debug_every == 0
    debug_reprojection = args.debug_reprojection and step % args.debug_every == 0
    noise_rng = None
    if is_gt_noisy_name(estimator):
        noise_std = gt_noise_std_m_from_name(estimator, args.gt_noise_std_m)
        noise_rng = np.random.default_rng(np.random.SeedSequence([
            int(args.gt_noise_seed), int(step), int(round(noise_std * 1_000_000_000))]))
    elif is_gt_d435_name(estimator):
        noise_rng = np.random.default_rng(np.random.SeedSequence([int(args.d435_noise_seed), int(step)]))
    fixed = reconstruct_fixed_pcd_for_estimator(
        est_name=estimator, imgs=imgs, depths_gt=depths, depth_estimator=depth_estimator,
        masks_u8=masks, intrinsics=intrinsics, extrinsics=extrinsics,
        resolution=args.resolution, mask_input=args.mask_input, num_points=args.num_points,
        use_cuda_fps=use_cuda_fps, debug=debug_reprojection,
        debug_name=f'{estimator} step {step}', do_scale_icp=not args.no_scale_icp,
        return_color=debug, gt_noise_std_m=args.gt_noise_std_m, noise_rng=noise_rng,
        d435_baseline_m=args.d435_baseline_m,
        d435_disparity_noise_std_px=args.d435_disparity_noise_std_px,
        d435_disparity_bias_px=args.d435_disparity_bias_px,
        d435_disparity_quantization_px=args.d435_disparity_quantization_px)
    if debug:
        pcd_stats(f'{estimator} step {step} fixed world', fixed)
        visualise_pcds([fixed], extrinsics, intrinsics, gt_pcd=None,
                      voxel_size=0.0, window_name=f'{estimator} step {step} fixed world cloud')
    # Preserve the collector's XYZ + rotation-vector state interpretation.
    transform = pose_6dof_to_transform(states[:6])
    xyz = transform_pcd(fixed[:, :3], np.linalg.inv(transform))
    if debug:
        colored = np.hstack((xyz, fixed[:, 3:6])) if fixed.shape[1] >= 6 else xyz
        pcd_stats(f'{estimator} step {step} fixed EEF/stored', colored)
    return xyz


def main(argv=None):
    parser = build_parser()
    try:
        args = validate_arguments(parser.parse_args(argv))
    except ValueError as error:
        parser.error(str(error))
    # mode='r' also ensures an absent/misspelled input path is never created.
    root = zarr.open_group(args.zarr_path, mode='r')
    if not isinstance(root.store, zarr.storage.DirectoryStore):
        raise ValueError('Use a local DirectoryStore/NestedDirectoryStore for transactional preprocessing.')
    data, intrinsics, steps, source = validate_input_schema(root, args.resolution, args.limit_steps)
    use_cuda_fps = not args.fps_cpu and torch.cuda.is_available()
    configs = {f'point_cloud_{estimator}': processing_config(args, estimator, source, use_cuda_fps)
               for estimator in sorted(args.estimators)}
    preflight_static_configuration(root, configs, args.num_points, steps, args.resume)
    # Completed publication can be recovered independently of model downloads.
    # This only finalizes an already computed transaction, never new inference.
    if not args.dry_run:
        recover_pending_publication(root)
    # All new reconstruction writes still follow successful model initialization
    # and a comparison of the actual loaded parameter/buffer identities.
    estimators = {name: make_depth_estimator(name) for name in args.estimators}
    for name, adapter in estimators.items():
        configs[f'point_cloud_{name}']['model_identity'] = loaded_model_identity(name, adapter)
    transaction = DerivedPCDTransaction(root, configs, args.num_points, steps, args.resume)
    transaction.preflight()
    cprint(f'[INFO] steps={steps}/{source["source_steps"]}, estimators={args.estimators}, '
           f'num_points={args.num_points}, dry_run={args.dry_run}', 'green')
    context = nullcontext(None) if args.dry_run else transaction
    with context as active:
        for step in tqdm(range(steps), desc='Reconstructing'):
            pending = [name for name in args.estimators if active is None
                       or step >= active.committed_steps(f'point_cloud_{name}')]
            if not pending:
                continue
            imgs = data['img'][step].astype(np.uint8)
            depths = data['depth'][step].astype(np.float32)
            masks = data['mask'][step]
            extrinsics = data['cam_extrinsics'][step].astype(np.float32)
            states = data['state'][step].astype(np.float32)
            for name in pending:
                row = reconstruct_step(args, name, estimators[name], step,
                    imgs, depths, masks, intrinsics, extrinsics, states, use_cuda_fps)
                if active is not None:
                    active.append(f'point_cloud_{name}', step, row)
        if active is not None:
            active.publish()
    cprint('[DONE] Dry run left the store unchanged.' if args.dry_run else
           '[DONE] All requested derived arrays were published successfully.', 'green')
    return {'steps': steps, 'estimators': args.estimators, 'dry_run': args.dry_run}


if __name__ == '__main__':
    main()
