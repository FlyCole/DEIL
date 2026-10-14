"""Run DEIL screening, paper Table I, and custom evaluations with public IP."""
from __future__ import annotations

import csv
import hashlib
import inspect
import json
import os
import pickle
from pathlib import Path
import random
import tempfile

from rlb_eval_config import (build_parser, resolve_experiment, experiment_to_dict,
                             evaluation_seed, EVALUATION_TASK_ORDER)
from public_ip import (load_public_policy, public_policy_provenance,
                       sample_to_cond_demo, predict_actions, subsample_pcd,
                       pose_to_transform, transform_to_pose, transform_pcd)

TASK_NAMES = {
    'lift_lid': 'TakeLidOffSaucepan', 'phone_on_base': 'PhoneOnBase',
    'open_box': 'OpenBox', 'slide_block': 'SlideBlockToTarget', 'close_box': 'CloseBox',
    'basketball': 'BasketballInHoop', 'buzz': 'BeatTheBuzz',
    'close_microwave': 'CloseMicrowave', 'plate_out': 'TakePlateOffColoredDishRack',
    'toilet_seat_down': 'ToiletSeatDown', 'toilet_seat_up': 'ToiletSeatUp',
    'toilet_roll_off': 'TakeToiletRollOffStand', 'open_microwave': 'OpenMicrowave',
    'lamp_on': 'LampOn', 'umbrella_out': 'TakeUmbrellaOutOfUmbrellaStand',
    'push_button': 'PushButton', 'put_rubbish': 'PutRubbishInBin',
    'close_jar': 'CloseJar', 'open_jar': 'OpenJar', 'meat_off_grill': 'MeatOffGrill',
}
EVALUATION_CAMERA_ORDER = ('front', 'front|left_shoulder',
                           'front|left_shoulder|wrist',
                           'front|left_shoulder|right_shoulder|wrist|overhead')
EVALUATION_RESULT_FIELDS = ('task_name', 'depth_estimator', 'camera_names',
                            'resolution', 'mask_input', 'avg_sr', 'std_sr', 'sr_list')
_RUNTIME_LOADED = False


def _load_runtime_dependencies():
    """Keep planning and --help independent of the simulator/model environment."""
    global _RUNTIME_LOADED
    if _RUNTIME_LOADED:
        return
    import cv2
    import numpy as np
    import open3d as o3d
    from scipy.spatial.transform import Rotation as Rot
    from tqdm import tqdm
    from rlbench.action_modes.action_mode import MoveArmThenGripper
    from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaIK
    from rlbench.action_modes.gripper_action_modes import Discrete
    from rlbench.environment import Environment, TaskEnvironment
    from rlbench.observation_config import ObservationConfig
    import rlbench.tasks as tasks
    from rlb_cameras import RLBCameras
    from pcd_scaler import scale_icp
    from utils import visualise_pcds
    import rlb_rollout_recorder as recording
    globals().update({k: v for k, v in locals().items()
                      if k not in {'tasks', 'recording'}})
    for name in ('RolloutArtifactRecorder', 'complete_manifest', 'fail_manifest',
                 'quarantine_orphan_videos', 'resume_manifest', 'rollout_outcomes',
                 'rollout_config_dir', 'single_eval_dir', 'start_manifest',
                 'validate_eval_artifacts'):
        globals()[name] = getattr(recording, name)
    for key, cls in tuple(TASK_NAMES.items()):
        if isinstance(cls, str):
            TASK_NAMES[key] = getattr(tasks, cls)
    _RUNTIME_LOADED = True


def set_random_seed(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_depth_estimator(name):
    if name == 'gt':
        return None
    if name in {'map_anything', 'map_anything_no_ext'}:
        from depth_estimators.map_anything_est import MapAnythingEst
        return MapAnythingEst(use_camera_poses=name == 'map_anything')
    modules = {'pi3': ('pi3_est', 'Pi3Est'), 'vggt': ('vggt_est', 'VGGT'),
               'moge': ('moge_depth', 'MoGeDepth'), 'mast3r': ('mast3r_est', 'Mast3rEst')}
    import importlib
    if name not in modules:
        raise ValueError(f'Unknown estimator: {name}')
    module, cls = modules[name]
    return getattr(importlib.import_module('depth_estimators.' + module), cls)()


def demo_cache_path(cache_dir, protocol, demo_seed, task_name, camera_names,
                    resolution, demo_iter, num_demos=1):
    # Sources/masking do not enter this key: every estimator shares the same raw demo.
    cameras = '-'.join(camera_names)
    return (Path(cache_dir) / protocol / f'seed_{demo_seed}' / task_name
            / cameras / str(resolution)
            / f'demo_{demo_iter:04d}_contexts_{num_demos}.pkl')


def save_raw_demo_cache(path, full_sample, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temp_path = Path(handle.name)
            pickle.dump({'schema_version': 1, 'metadata': metadata,
                         'sample': {'demos': full_sample['demos'], 'live': {}}},
                        handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def load_raw_demo_cache(path, expected_metadata):
    with open(path, 'rb') as handle:
        payload = pickle.load(handle)
    if payload.get('schema_version') != 1 or payload.get('metadata') != expected_metadata:
        raise ValueError(f'Raw-demo cache metadata does not match: {path}')
    sample = payload['sample']
    if len(sample['demos']) != expected_metadata['num_demos']:
        raise ValueError(f'Raw-demo cache count does not match: {path}')
    return sample


def get_demos(task, num_demos):
    _load_runtime_dependencies()
    demos = []
    for _ in range(num_demos):
        # A persistent simulator error must propagate rather than retry forever.
        demos.append(task.get_demos(1, live_demos=True, max_attempts=1000)[0])
    return {'demos': demos, 'live': {}}


def _recording_code_digest(depth_estimator=None):
    names = ('rlb_task_eval.py', 'public_ip.py', 'rlb_eval_config.py',
             'rlb_rollout_recorder.py', 'rlb_cameras.py', 'pcd_scaler.py',
             'utils.py', 'depth_estimators/base_estimator.py',
             'depth_estimators/dependency_paths.py', 'depth_dependencies.json',
             'depth_estimators/pi3_est.py', 'depth_estimators/vggt_est.py',
             'depth_estimators/moge_depth.py', 'depth_estimators/mast3r_est.py',
             'depth_estimators/map_anything_est.py')
    # Include adapters before model loading, including CSV-only resume mode.
    paths = [Path(__file__).parent / name for name in names]
    if depth_estimator is not None:
        source = inspect.getsourcefile(type(depth_estimator))
        if source:
            paths.append(Path(source))
    hasher = hashlib.sha256()
    for path in sorted(set(paths)):
        hasher.update(path.name.encode())
        hasher.update(path.read_bytes())
    return hasher.hexdigest()


def rollout_recording_metadata(task_name, depth_est_name, camera_names, resolution,
                               mask_input, use_gt, num_demos, num_demo_iter,
                               num_rollouts, video_fps, depth_estimator=None,
                               protocol='custom', demo_seed=500_000_000,
                               rollout_seed=600_000_000, experiment_id=None,
                               checkpoint_path=None):
    return dict(task_name=task_name, depth_estimator=depth_est_name,
                camera_names=list(camera_names), resolution=int(resolution),
                mask_input=None if use_gt else bool(mask_input), num_demos=int(num_demos),
                num_demo_iterations=int(num_demo_iter), num_rollouts_per_demo=int(num_rollouts),
                protocol=protocol, demo_seed=demo_seed, rollout_seed=rollout_seed,
                experiment_id=experiment_id, max_execution_steps=30, execution_horizon=8,
                num_policy_points=2048, point_cloud_frame='eef',
                agent_pos_format='xyz_rotvec_gripper',
                rgb_camera='front' if 'front' in camera_names else camera_names[0],
                state_sampling='policy_decision', rgb_sampling='environment_step',
                state_video_alignment='decision_video_frame_indices', video_fps=float(video_fps),
                recording_code_sha256=_recording_code_digest(depth_estimator),
                public_policy=public_policy_provenance(checkpoint_path))


def process_demos(task, full_sample, depth_estimator, num_waypoints=10,
                  num_points=2048, camera_names=('front', 'left_shoulder'),
                  resolution=256, mask_input=False, use_gt=False, debug=False):
    _load_runtime_dependencies()
    processed = []
    for demo in full_sample['demos']:
        poses = [pose_to_transform(obs.gripper_pose) for obs in demo]
        grips = [obs.gripper_open for obs in demo]
        indices = extract_waypoints(np.asarray(poses), np.asarray(grips), num_waypoints)
        # Estimate only selected waypoints, then use the published IP conditioning API.
        sample = {'pcds': [], 'T_w_es': [poses[i] for i in indices],
                  'grips': [grips[i] for i in indices]}
        for i in indices:
            sample['pcds'].append(get_point_cloud(task, demo[i], depth_estimator,
                                                  camera_names=camera_names,
                                                  resolution=resolution, mask_input=mask_input,
                                                  use_gt=use_gt, debug=debug))
        processed.append(sample_to_cond_demo(sample, num_waypoints, num_points))
    return processed


def initialise_rlbench(task_name='phone_on_base',
                       camera_names=['front', 'left_shoulder', 'right_shoulder', 'wrist'],
                       resolution=128,
                       headless=False,
                       restrict_rot=True):
    _load_runtime_dependencies()
    obs_config = ObservationConfig()
    obs_config.set_all_low_dim(True)
    obs_config.set_all_high_dim(False)

    for cam_name in camera_names:
        cam = getattr(obs_config, f'{cam_name}_camera')
        cam.set_all(True)
        cam.depth_in_meters = True


    action_mode = MoveArmThenGripper(
        arm_action_mode=EndEffectorPoseViaIK(),
        gripper_action_mode=Discrete()
    )
    env = Environment(action_mode,
                      './',
                      obs_config=obs_config,
                      headless=headless)
    env.launch()
    task = env.get_task(TASK_NAMES[task_name])

    def temp(position, euler=None, quaternion=None, ignore_collisions=False, trials=300, max_configs=1,
             distance_threshold=0.65, max_time_ms=10, trials_per_goal=1, algorithm=None, relative_to=None):
        return env._robot.arm.get_linear_path(position,
                                              euler,
                                              quaternion,
                                              ignore_collisions=ignore_collisions,
                                              relative_to=relative_to)

    env._robot.arm.get_path = temp
    env._scene._start_arm_joint_pos = np.array([6.74760377e-05, -1.91104114e-02, -3.62065766e-05, -1.64271665e+00,
                                                -1.14094291e-07, 1.55336857e+00, 7.85427451e-01])

    rot_bounds = env._scene.task.base_rotation_bounds()
    mean_rot = (rot_bounds[0][2] + rot_bounds[1][2]) / 2
    if restrict_rot:
        env._scene.task.base_rotation_bounds = lambda: ((0.0, 0.0, max(rot_bounds[0][2], mean_rot - np.pi / 3)),
                                                        (0.0, 0.0, min(rot_bounds[1][2], mean_rot + np.pi / 3)))

    for camera_name in camera_names:
        if camera_name == 'front':
            cam = task._scene._cam_front
            cam_mask = task._scene._cam_front_mask
        elif camera_name == 'left_shoulder':
            cam = task._scene._cam_over_shoulder_left
            cam_mask = task._scene._cam_over_shoulder_left_mask
            cam.set_far_clipping_plane(4.5)
            cam_mask.set_far_clipping_plane(4.5)
            cam.set_pose([0.2, 1.3, 1.5, 0.819, 0., 0., 0.574])
            cam_mask.set_pose([0.2, 1.3, 1.5, 0.819, 0., 0., 0.574])
        elif camera_name == 'right_shoulder':
            cam = task._scene._cam_over_shoulder_right
            cam_mask = task._scene._cam_over_shoulder_right_mask
        elif camera_name == 'wrist':
            cam = task._scene._cam_wrist
            cam_mask = task._scene._cam_wrist_mask
            cam.set_far_clipping_plane(6)
            cam_mask.set_far_clipping_plane(6)

        elif camera_name == 'overhead':
            cam = task._scene._cam_overhead
            cam_mask = task._scene._cam_overhead_mask
        else:
            raise ValueError(f'Unknown camera: {camera_name}')

        cam.set_resolution([resolution, resolution])
        cam_mask.set_resolution([resolution, resolution])


    return env, task


def evaluation_result_sort_key(row):
    """Sort result rows by task, camera setup, resolution, then mask."""
    task_name = str(row.get('task_name', ''))
    camera_names = str(row.get('camera_names', ''))
    try:
        resolution = int(row.get('resolution', 0))
    except (TypeError, ValueError):
        resolution = 0
    mask_name = normalize_bool(row.get('mask_input'))
    task_rank = {
        name: index for index, name in enumerate(EVALUATION_TASK_ORDER)
    }
    camera_rank = {
        name: index for index, name in enumerate(EVALUATION_CAMERA_ORDER)
    }
    mask_rank = {'none': 0, 'false': 1, 'true': 2}
    return (
        task_rank.get(task_name, len(task_rank)),
        task_name.casefold(),
        camera_rank.get(camera_names, len(camera_rank)),
        camera_names.casefold(),
        resolution,
        mask_rank.get(mask_name, len(mask_rank)),
        mask_name,
    )


def _write_evaluation_result_rows(csv_path, rows):
    """Atomically write evaluation rows in deterministic configuration order."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    ordered_rows = sorted(rows, key=evaluation_result_sort_key)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='w',
            encoding='utf-8',
            newline='',
            dir=csv_path.parent,
            prefix=f'.{csv_path.name}.',
            suffix='.tmp',
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
            writer = csv.DictWriter(
                temp_file,
                fieldnames=EVALUATION_RESULT_FIELDS,
            )
            writer.writeheader()
            writer.writerows(ordered_rows)
        os.replace(temp_path, csv_path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def sort_evaluation_result_csv(csv_path):
    """Reorder an existing evaluation CSV without changing its rows."""
    csv_path = Path(csv_path)
    with csv_path.open('r', encoding='utf-8', newline='') as csv_file:
        rows = list(csv.DictReader(csv_file))
    _write_evaluation_result_rows(csv_path, rows)
    return len(rows)


def append_evaluation_result(csv_path, row):
    """Append one evaluation result while preserving deterministic ordering."""
    csv_path = Path(csv_path)
    rows = []
    if csv_path.exists():
        with csv_path.open('r', encoding='utf-8', newline='') as csv_file:
            rows.extend(csv.DictReader(csv_file))
    rows.append(row)
    _write_evaluation_result_rows(csv_path, rows)


def erode_rlbench_mask(mask, resolution):
    """Apply the segmentation-mask erosion used for policy point clouds."""
    if resolution == 512:
        kernel = np.ones((5, 5), np.uint8)
    elif resolution == 256:
        kernel = np.ones((3, 3), np.uint8)
    else:
        kernel = np.ones((2, 2), np.uint8)
    return cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def get_gt_reference_point_cloud(obs, camera_names, resolution):
    """Return the finite, segmented simulator point cloud in world frame."""
    point_clouds = []
    for name in camera_names:
        point_cloud = np.asarray(
            getattr(obs, f'{name}_point_cloud')
        ).reshape(-1, 3)
        mask = erode_rlbench_mask(
            getattr(obs, f'{name}_mask') > 60,
            resolution,
        ).reshape(-1)
        valid = mask & np.all(np.isfinite(point_cloud), axis=1)
        point_clouds.append(point_cloud[valid])

    nonempty = [point_cloud for point_cloud in point_clouds
                if point_cloud.shape[0] != 0]
    if not nonempty:
        return np.zeros((0, 3), dtype=np.float32)
    return np.concatenate(nonempty, axis=0)


def dp3_agent_position(obs, T_w_e):
    """Return DP3's [xyz, rotation-vector, gripper] robot state."""
    rotvec = Rot.from_matrix(T_w_e[:3, :3]).as_rotvec()
    return np.concatenate([
        T_w_e[:3, 3],
        rotvec,
        [obs.gripper_open],
    ]).astype(np.float32)


def demo_resume_checkpoint_path(eval_dir, demo_iter):
    return Path(eval_dir) / 'resume_demos' / f'demo_{int(demo_iter):04d}.pkl'


def save_demo_resume_checkpoint(eval_dir, demo_iter, full_sample):
    """Atomically retain processed demo conditioning for rollout resumption."""
    checkpoint_path = demo_resume_checkpoint_path(eval_dir, demo_iter)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f'.{checkpoint_path.stem}.',
            suffix='.tmp',
            dir=checkpoint_path.parent,
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
            pickle.dump(
                {'demos': full_sample['demos'], 'live': {}},
                temp_file,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        os.replace(temp_path, checkpoint_path)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def load_demo_resume_checkpoint(eval_dir, demo_iter):
    checkpoint_path = demo_resume_checkpoint_path(eval_dir, demo_iter)
    with checkpoint_path.open('rb') as checkpoint_file:
        full_sample = pickle.load(checkpoint_file)
    if (
        not isinstance(full_sample, dict)
        or 'demos' not in full_sample
        or 'live' not in full_sample
    ):
        raise ValueError(f'Invalid demo resume checkpoint: {checkpoint_path}')
    return full_sample


def get_point_cloud(task,
                    obs,
                    depth_estimator,
                    previous_scale=None,
                    camera_names=['front', 'left_shoulder', 'right_shoulder', 'wrist'],
                    mask_input: bool = False,
                    resolution: int = 128,
                    use_gt: bool = False,
                    debug: bool = False):
    if depth_estimator is None:
        use_gt = True

    intrinsics = []
    extrinsics = []
    depths = []
    rgbs = []
    masks = []
    gt_pcds = []
    gt_masked_pcd = []

    for name in camera_names:
        gt_pcds.append(getattr(obs, f'{name}_point_cloud').reshape(-1, 3))
        masks.append(getattr(obs, f'{name}_mask') > 60)

        if use_gt:
            depths.append(getattr(obs, f'{name}_depth'))
        else:
            rgbs.append(getattr(obs, f'{name}_rgb'))

        if name == 'front':
            camera = task._scene._cam_front
        elif name == 'left_shoulder':
            camera = task._scene._cam_over_shoulder_left
        elif name == 'right_shoulder':
            camera = task._scene._cam_over_shoulder_right
        elif name == 'wrist':
            camera = task._scene._cam_wrist
        elif name == 'overhead':
            camera = task._scene._cam_overhead

        intrinsics.append(camera.get_intrinsic_matrix())
        if name == 'wrist':
            obs_misc = getattr(obs, 'misc', None)
            wrist_extrinsics = (
                obs_misc.get('wrist_camera_extrinsics')
                if isinstance(obs_misc, dict)
                else None
            )
            if wrist_extrinsics is None:
                wrist_extrinsics = getattr(obs, 'wrist_matrix', None)
            if wrist_extrinsics is None:
                raise AttributeError(
                    'Wrist observation has neither '
                    'misc["wrist_camera_extrinsics"] nor legacy '
                    'wrist_matrix extrinsics.'
                )
            extrinsics.append(np.asarray(wrist_extrinsics, dtype=np.float64))
        else:
            extrinsics.append(camera.get_matrix())

    # Visualise rgbs and masked rgbs
    if mask_input:
        masked_rgbs = []
        for i in range(len(rgbs)):
            masked_rgb = rgbs[i].copy()
            masked_rgb[~masks[i]] = 0
            masked_rgbs.append(masked_rgb)


    if not use_gt:
        if mask_input:
            est_data = depth_estimator.infer(masked_rgbs, intrinsics, extrinsics)
        else:
            est_data = depth_estimator.infer(rgbs, intrinsics, extrinsics)

        if 'depth' in est_data.keys():
            depths = [data for data in est_data['depth']]
        elif 'pts3d' in est_data.keys():
            pcd = est_data['pts3d']
            visualise_pcds([pcd], extrinsics, intrinsics, gt_pcds)


    # Mask erosion to remove edges
    for i, mask in enumerate(masks):
        masks[i] = erode_rlbench_mask(mask, resolution)

    repr = RLBCameras.reproject_to_pcd
    if len(depths) > len(intrinsics):
        depths = depths[:len(intrinsics)]
    reprojected_pcds = [repr(depths[i], intrinsics[i], extrinsics[i], masks[i]) for i in range(len(depths))]

    if debug:
        visualise_pcds(reprojected_pcds, extrinsics, intrinsics, gt_pcds)

    # Outlier removal for specific depth estimators
    pcd_eliminated = False
    if depth_estimator is not None and depth_estimator.name in {'map_anything', 'map_anything_no_ext'}:
        for i, pcd in enumerate(reprojected_pcds):
            pcd = np.array(pcd, dtype=np.float64, copy=True, order="C")
            pcd_noisy = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd))
            nb_neighbors = int(70 * resolution ** 2 / 700 ** 2)
            pcd_sor, ind_sor = pcd_noisy.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=2.0)
            reprojected_pcds[i] = np.asarray(pcd_sor.points)
            if reprojected_pcds[i].shape[0] == 0:
                pcd_eliminated = True

    if use_gt or len(reprojected_pcds) == 1 or pcd_eliminated:
        scaled_pcds = reprojected_pcds
        return np.concatenate(scaled_pcds, axis=0)
    else:
        try:
            scaled_pcds, best_scale = scale_icp(reprojected_pcds,
                                                extrinsics,
                                                previous_scale=None,
                                                n_initialisations=5,
                                                max_iterations=100,
                                                intrinsics=(task._scene._cam_front.get_resolution(), intrinsics[0]),
                                                visualise=False)
        except Exception as error:
            raise RuntimeError('Point-cloud scale alignment failed') from error

    # Visualise unscaled pcds
    if debug:
        visualise_pcds(scaled_pcds, extrinsics, intrinsics, gt_pcds)

    pruned_pcds = []
    for scale_idx in range(len(scaled_pcds)):
        if scaled_pcds[scale_idx].shape[0] != 0:
            pruned_pcds.append(scaled_pcds[scale_idx])

    return np.concatenate(pruned_pcds, axis=0)


def extract_waypoints(traj, traj_states, num_waypoints):
    waypoints = [0, len(traj) - 1]
    # Add all waypoints where traj state changes.
    for i in range(1, len(traj_states) - 2):
        if traj_states[i] != traj_states[i - 1] and i not in waypoints:
            waypoints.append(i)
    waypoints.sort()

    for i in range(1, len(traj_states)):
        err = pose_error(traj[i], traj[i - 1], rot_scale=1)
        closest_waypoint = np.argmin([abs(i - w) for w in waypoints])
        if err < 3.5e-3 and abs(i - waypoints[closest_waypoint]) > 5:
            waypoints.append(i)

    waypoints.sort()
    wp_to_add = num_waypoints - len(waypoints)

    wp_segment_dist = [pose_error(traj[waypoints[i]], traj[waypoints[i + 1]], rot_scale=1) for i in
                       range(len(waypoints) - 1)]
    wp_segment_dist = np.array(wp_segment_dist) / np.sum(wp_segment_dist)
    extra_wp_segments = np.ceil(wp_segment_dist * wp_to_add).astype(int)

    seg_order = np.argsort(extra_wp_segments)[::-1]
    for ii in range(len(extra_wp_segments)):
        i = seg_order[ii]
        wp_a = waypoints[i]
        wp_b = waypoints[i + 1]
        for j in range(extra_wp_segments[i]):
            waypoints.append(wp_a + (wp_b - wp_a) * (j + 1) // (extra_wp_segments[i] + 1))
            if len(waypoints) == num_waypoints:
                break
        else:
            continue
        break

    waypoints.sort()
    return waypoints


def pose_error(T1, T2, rot_scale=0.01):
    dist_trans = np.linalg.norm(T1[:3, 3] - T2[:3, 3])
    dist_rot = Rot.from_matrix(T1[:3, :3] @ T2[:3, :3].T).magnitude()
    return dist_trans + rot_scale * dist_rot


def normalize_bool(x):
    if x is None or str(x).strip().lower() in {'', 'none', 'null'}:
        return "none"
    return str(x).strip().lower()


def evaluation_done(csv_path, task_name, depth_est_name, camera_names, resolution, mask_input):
    if not os.path.exists(csv_path):
        return False

    camera_key = "|".join(camera_names)
    mask_key = normalize_bool(mask_input)

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (
                row["task_name"] == task_name
                and row["depth_estimator"] == depth_est_name
                and row["camera_names"] == camera_key
                and row["resolution"] == str(resolution)
                and normalize_bool(row["mask_input"]) == mask_key
            ):
                if row["avg_sr"] and row["std_sr"] and row["sr_list"]:
                    return True

    return False


def evaluate_agent(task,
                   model,
                   depth_estimator,
                   full_sample,
                   max_execution_steps=30,
                   execution_horizon=8,
                   num_rollouts=2,
                   camera_names=['front', 'left_shoulder', 'right_shoulder', 'wrist'],
                   mask_input: bool = False,
                   resolution: int = 128,
                   use_gt: bool = False,
                   debug: bool = False,
                   artifact_eval_dir=None,
                   artifact_task_name: str = None,
                   artifact_depth_est_name: str = None,
                   artifact_demo_iter: int = 0,
                   rollout_id_offset: int = 0,
                   artifact_num_points: int = 2048,
                   video_fps: float = 10.0,
                   rollout_indices=None,
                   rollout_seed=200_000_000):
    if rollout_indices is None:
        rollout_indices = tuple(range(num_rollouts))
    else:
        rollout_indices = tuple(int(index) for index in rollout_indices)
        if len(set(rollout_indices)) != len(rollout_indices):
            raise ValueError('rollout_indices contains duplicate IDs')
        if any(not 0 <= index < num_rollouts for index in rollout_indices):
            raise ValueError(
                f'rollout_indices must be within 0..{num_rollouts - 1}'
            )

    successes = []
    scheduled_rollouts = len(rollout_indices)
    pbar = tqdm(
        rollout_indices,
        desc=f'Evaluating model, SR: 0/{scheduled_rollouts}',
        total=scheduled_rollouts,
        leave=False,
    )
    for rollout_in_demo in pbar:
        set_random_seed(evaluation_seed(rollout_seed, artifact_task_name, artifact_demo_iter, rollout_in_demo))
        max_reset_tries = 1000
        for count in range(1, max_reset_tries + 1):
            try:
                task.reset()
                break
            except Exception as e:
                print(f"Reset failed {count}/{max_reset_tries}: {repr(e)}")
        else:
            raise RuntimeError("task.reset() failed too many times")

        # pcd_vis = o3d.visualization.Visualizer()
        env_action = np.zeros(8)
        # number of steps in rollouts.
        success = 0

        recorder = None
        if artifact_eval_dir is not None:
            video_camera = 'front' if 'front' in camera_names else camera_names[0]
            recorder = RolloutArtifactRecorder(
                eval_dir=artifact_eval_dir,
                rollout_id=rollout_id_offset + rollout_in_demo,
                demo_iter=artifact_demo_iter,
                rollout_in_demo=rollout_in_demo,
                task_name=artifact_task_name or 'unknown',
                depth_estimator=artifact_depth_est_name or (
                    'gt' if use_gt else getattr(depth_estimator, 'name', 'unknown')
                ),
                camera_names=camera_names,
                resolution=resolution,
                mask_input=mask_input,
                num_points=artifact_num_points,
                fps=video_fps,
                video_camera=video_camera,
            )

        for k in range(max_execution_steps):
            curr_obs = task.get_observation()
            T_w_e = pose_to_transform(curr_obs.gripper_pose)
            pcd = get_point_cloud(task,
                                  curr_obs,
                                  depth_estimator,
                                  camera_names=camera_names,
                                  resolution=resolution,
                                  mask_input=mask_input,
                                  use_gt=use_gt,
                                  debug=debug)
            subsampled_pcd = subsample_pcd(pcd, artifact_num_points)
            policy_pcd_eef = transform_pcd(
                subsampled_pcd,
                np.linalg.inv(T_w_e),
            ).astype(np.float32)
            full_sample['live']['obs'] = [policy_pcd_eef]
            full_sample['live']['grips'] = [curr_obs.gripper_open]
            full_sample['live']['T_w_es'] = [T_w_e]

            if recorder is not None:
                if use_gt:
                    # This mirrors DP3: for a GT policy, its observation is also
                    # the metric reference rather than a second random sample.
                    gt_pcd_eef = policy_pcd_eef.copy()
                else:
                    # GT sampling is analysis-only. Preserve NumPy RNG state so
                    # enabling recording cannot change later policy samples.
                    rng_state = np.random.get_state()
                    try:
                        gt_pcd_world = get_gt_reference_point_cloud(
                            curr_obs,
                            camera_names,
                            resolution,
                        )
                        if gt_pcd_world.shape[0] == 0:
                            raise RuntimeError(
                                'Cannot record GT reference: all requested '
                                'camera masks are empty or non-finite.'
                            )
                        gt_pcd_eef = transform_pcd(
                            subsample_pcd(gt_pcd_world, artifact_num_points),
                            np.linalg.inv(T_w_e),
                        ).astype(np.float32)
                    finally:
                        np.random.set_state(rng_state)

                recorder.record_decision(
                    agent_pos=dp3_agent_position(curr_obs, T_w_e),
                    T_w_e=T_w_e,
                    policy_point_cloud=policy_pcd_eef,
                    gt_point_cloud=gt_pcd_eef,
                    rgb_frame=getattr(curr_obs, f'{recorder.video_camera}_rgb'),
                )


            actions, grips = predict_actions(model, full_sample)

            for j in range(min(execution_horizon, len(actions))):
                env_action[:7] = transform_to_pose(T_w_e @ actions[j])
                env_action[7] = int((grips[j] + 1) / 2 > 0.5)
                try:
                    curr_obs, reward, terminate = task.step(env_action)
                    success = int(terminate and reward > 0.)
                except Exception as e:
                    print(f"Step failed at step {k}, action {j}: {repr(e)}")
                    terminate = True
                else:
                    # Keep recorder failures outside the simulator exception
                    # path so they fail the artifact run instead of changing
                    # the rollout's task outcome.
                    if recorder is not None:
                        recorder.record_video_frame(
                            getattr(curr_obs, f'{recorder.video_camera}_rgb')
                        )
                if terminate:
                    break

            else:
                continue
            break

        if recorder is not None:
            recorder.save(bool(success))
        successes.append(success)
        pbar.set_description(f'Evaluating model, SR: {sum(successes)}/{len(successes)}')
        pbar.refresh()


    pbar.close()
    return sum(successes) / len(successes)


def run_evaluation(task_name: str,
                   model,
                   depth_estimator,
                   depth_est_name: str,
                   num_demos: int = 1,
                   num_demo_iter: int = 20,
                   num_rollouts: int = 5,
                   headless: bool = True,
                   camera_names: list = ['front', 'left_shoulder', 'right_shoulder', 'wrist'],
                   resolution: int = 128,
                   mask_input: bool = False,
                   load_cached_demo: bool = False,
                   use_gt: bool = False,
                   debug: bool = False,
                   record_rollouts: bool = False,
                   rollout_output_dir: str = 'results/ip_eval',
                   video_fps: float = 10.0,
                   resume: bool = False,
                   protocol: str = "custom",
                   demo_seed: int = 500_000_000,
                   rollout_seed: int = 600_000_000,
                   demo_cache_dir: str = "demos_ip",
                   experiment_id: str = None,
                   checkpoint_path: str = None):

    _load_runtime_dependencies()
    artifact_eval_dir = None
    expected_rollouts = int(num_demo_iter * num_rollouts)
    completed_outcomes = {}
    if record_rollouts:
        artifact_config_dir = rollout_config_dir(
            rollout_output_dir,
            task_name,
            depth_est_name,
            camera_names,
            resolution,
            None if use_gt else mask_input,
        )
        artifact_metadata = rollout_recording_metadata(
            task_name,
            depth_est_name,
            camera_names,
            resolution,
            mask_input,
            use_gt,
            num_demos,
            num_demo_iter,
            num_rollouts,
            video_fps,
            depth_estimator,
            protocol=protocol, demo_seed=demo_seed, rollout_seed=rollout_seed,
            experiment_id=experiment_id, checkpoint_path=checkpoint_path,
        )
        artifact_eval_dir = single_eval_dir(artifact_config_dir, create=True)
        artifact_paths_exist = (
            any((artifact_eval_dir / 'live_states').glob('rollout_*.npz'))
            or any((artifact_eval_dir / 'live_videos').glob('rollout_*.mp4'))
        )
        manifest_path = artifact_eval_dir / 'manifest.json'
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text())
            mismatches = [key for key, value in artifact_metadata.items()
                          if existing.get(key) != value]
            if mismatches:
                raise RuntimeError('Incompatible rollout manifest: ' + ', '.join(mismatches))
        completed_outcomes = rollout_outcomes(
            artifact_eval_dir,
            expected_rollouts=expected_rollouts,
            allow_orphan_videos=resume,
        )
        if (
            not resume
            and artifact_paths_exist
            and len(completed_outcomes) != expected_rollouts
        ):
            raise RuntimeError(
                f'{artifact_eval_dir} contains a partial rollout set; enable '
                'rollout resumption to evaluate only its missing rollout IDs.'
            )

        quarantined_videos = ()
        if resume:
            quarantined_videos = quarantine_orphan_videos(artifact_eval_dir)
        manifest_path = artifact_eval_dir / 'manifest.json'
        manifest_status = None
        if manifest_path.exists():
            try:
                with manifest_path.open('r', encoding='utf-8') as manifest_file:
                    manifest_status = json.load(manifest_file).get('status')
            except (OSError, ValueError, TypeError):
                manifest_status = None

        if resume and manifest_status in {'running', 'failed'}:
            resume_manifest(artifact_eval_dir)
        elif not (
            manifest_status == 'complete'
            and len(completed_outcomes) == expected_rollouts
        ):
            start_manifest(artifact_eval_dir, artifact_metadata)

        if resume or completed_outcomes:
            print(
                f'Using rollout-level progress in {artifact_eval_dir}: '
                f'{len(completed_outcomes)}/{expected_rollouts} rollout pairs '
                'complete'
            )
            if quarantined_videos:
                print(
                    f'Moved {len(quarantined_videos)} incomplete video(s) to '
                    f'{artifact_eval_dir / "incomplete_artifacts"}.'
                )
        else:
            print(f'Recording rollout artifacts to {artifact_eval_dir}')

    env = None
    sr_list = []
    try:
        for i in range(num_demo_iter):
            rollout_id_offset = i * num_rollouts
            missing_rollouts = [
                rollout_in_demo
                for rollout_in_demo in range(num_rollouts)
                if rollout_id_offset + rollout_in_demo not in completed_outcomes
            ]
            if record_rollouts and not missing_rollouts:
                demo_outcomes = [
                    completed_outcomes[rollout_id_offset + rollout_in_demo]
                    for rollout_in_demo in range(num_rollouts)
                ]
                sr = float(np.mean(demo_outcomes))
                sr_list.append(sr)
                print(
                    f'Iteration {i + 1}/{num_demo_iter} already complete; '
                    f'SR: {sr:.3f}, Avg SR: {np.mean(sr_list):.3f} '
                    f'± {np.std(sr_list):.3f}'
                )
                continue

            # Re-initialise each iteration to prevent memory leaks.
            if env is not None:
                env.shutdown()
                env = None
            set_random_seed(evaluation_seed(demo_seed, task_name, i))
            env, task = initialise_rlbench(task_name=task_name,
                                           camera_names=camera_names,
                                           resolution=resolution,
                                           restrict_rot=True,
                                           headless=headless)

            resume_demo_path = None
            if artifact_eval_dir is not None:
                resume_demo_path = demo_resume_checkpoint_path(
                    artifact_eval_dir,
                    i,
                )
            if resume_demo_path is not None and resume_demo_path.exists():
                print(f'Loading resume demo checkpoint from {resume_demo_path} ...')
                full_sample = load_demo_resume_checkpoint(artifact_eval_dir, i)
            else:
                if resume and any(
                    rollout_id_offset + rollout_in_demo in completed_outcomes
                    for rollout_in_demo in range(num_rollouts)
                ):
                    print(
                        'Warning: this older partial iteration has no processed '
                        'demo checkpoint; its missing rollouts will use a '
                        'reconstructed demo.'
                    )
                cache_metadata = dict(protocol=protocol, demo_seed=demo_seed,
                                      task_name=task_name, camera_names=list(camera_names),
                                      resolution=resolution, demo_iter=i, num_demos=num_demos)
                cache_path = demo_cache_path(demo_cache_dir, protocol, demo_seed,
                                             task_name, camera_names, resolution, i, num_demos)
                if cache_path.exists():
                    full_sample = load_raw_demo_cache(cache_path, cache_metadata)
                elif load_cached_demo:
                    raise FileNotFoundError(f'No matching raw-demo cache: {cache_path}. '
                                            'Run rlb_demo_collection_ip.py with the same protocol first.')
                else:
                    set_random_seed(evaluation_seed(demo_seed, task_name, i))
                    full_sample = get_demos(task, num_demos)
                    save_raw_demo_cache(cache_path, full_sample, cache_metadata)
                set_random_seed(evaluation_seed(demo_seed, task_name, i))
                processed_demos = process_demos(task,
                                                full_sample,
                                                depth_estimator,
                                                num_waypoints=10,
                                                resolution=resolution,
                                                camera_names=camera_names,
                                                mask_input=mask_input,
                                                use_gt=use_gt,
                                                debug=debug)
                full_sample['demos'] = processed_demos
                if artifact_eval_dir is not None:
                    save_demo_resume_checkpoint(
                        artifact_eval_dir,
                        i,
                        full_sample,
                    )

            sr = evaluate_agent(task,
                                model,
                                depth_estimator,
                                full_sample,
                                num_rollouts=num_rollouts,
                                camera_names=camera_names,
                                mask_input=mask_input,
                                resolution=resolution,
                                use_gt=use_gt,
                                debug=debug,
                                artifact_eval_dir=artifact_eval_dir,
                                artifact_task_name=task_name,
                                artifact_depth_est_name=depth_est_name,
                                artifact_demo_iter=i,
                                rollout_id_offset=rollout_id_offset,
                                artifact_num_points=2048,
                                video_fps=video_fps,
                                rollout_indices=(
                                    missing_rollouts if record_rollouts else None
                                ), rollout_seed=rollout_seed)
            if record_rollouts:
                completed_outcomes = rollout_outcomes(
                    artifact_eval_dir,
                    expected_rollouts=expected_rollouts,
                )
                demo_outcomes = [
                    completed_outcomes[rollout_id_offset + rollout_in_demo]
                    for rollout_in_demo in range(num_rollouts)
                ]
                sr = float(np.mean(demo_outcomes))
            sr_list.append(sr)
            print(f'Iteration {i + 1}/{num_demo_iter}, SR: {sr:.3f}, Avg SR: {np.mean(sr_list):.3f} ± {np.std(sr_list):.3f}')
    except Exception as error:
        if artifact_eval_dir is not None:
            fail_manifest(artifact_eval_dir, repr(error))
        raise
    finally:
        if env is not None:
            env.shutdown()

    avg_sr = np.mean(sr_list)
    std_sr = np.std(sr_list)
    print(f'Final Results over {num_demo_iter * num_rollouts} rollouts: Avg SR: {avg_sr:.3f} ± {std_sr:.3f}')


    # Save results to csv
    output_name = str(Path(rollout_output_dir) / f'{depth_est_name}.csv')
    ss = {
        "task_name": task_name,
        "depth_estimator": depth_est_name,
        "camera_names": "|".join(camera_names) if isinstance(camera_names, (list, tuple)) else camera_names,
        "resolution": resolution,
        "mask_input": mask_input if not use_gt else None,
        "avg_sr": avg_sr,
        "std_sr": std_sr,
        "sr_list": "|".join(map(str, sr_list))  # convert list to string
    }

    # Rebuilding a result after artifact resumption replaces the same case row.
    if Path(output_name).exists():
        with open(output_name, newline='') as handle:
            rows = list(csv.DictReader(handle))
        rows = [row for row in rows if not (
            row['task_name'] == task_name and row['depth_estimator'] == depth_est_name
            and row['camera_names'] == ss['camera_names']
            and int(row['resolution']) == resolution
            and normalize_bool(row['mask_input']) == normalize_bool(ss['mask_input']))]
        rows.append(ss)
        _write_evaluation_result_rows(output_name, rows)
    else:
        append_evaluation_result(output_name, ss)

    print("=" * 130)

    if artifact_eval_dir is not None:
        complete_manifest(artifact_eval_dir, {
            'avg_sr': float(avg_sr),
            'std_sr': float(std_sr),
            'sr_list': [float(sr) for sr in sr_list],
            'recorded_rollouts': int(num_demo_iter * num_rollouts),
        })



def main(argv=None):
    parser = build_parser()
    parser.add_argument('--checkpoint_path', type=Path,
                        help='Public IP model.pt (default: instant_policy/checkpoints/model.pt).')
    parser.add_argument('--device', default='cuda', choices=('cpu', 'cuda'))
    parser.add_argument('--diffusion_steps', default=4, type=int)
    args = parser.parse_args(argv)
    try:
        config = resolve_experiment(args)
    except ValueError as exc:
        parser.error(str(exc))
    if config.compile_models:
        parser.error('The public IP deployment API does not expose model compilation.')
    if args.diffusion_steps <= 0:
        parser.error('--diffusion_steps must be positive')
    snapshot = experiment_to_dict(config)
    provenance = public_policy_provenance(args.checkpoint_path)
    identity = {'experiment_id': config.experiment_id,
                'public_native_sha256': provenance['native_module_sha256'],
                'checkpoint_sha256': provenance['checkpoint'].get('sha256'),
                'device': args.device, 'diffusion_steps': args.diffusion_steps,
                'recording_code_sha256': _recording_code_digest()}
    run_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    output_dir = Path(config.rollout_output_dir) / config.protocol / run_id
    snapshot.update(public_policy=provenance, device=args.device,
                    diffusion_steps=args.diffusion_steps,
                    execution_id=run_id, resolved_output_dir=str(output_dir))
    if config.dry_run:
        print(json.dumps(snapshot, indent=2))
        return snapshot
    model = load_public_policy(config.num_demos, args.device, args.checkpoint_path,
                               args.diffusion_steps)
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = output_dir / 'experiment.json'
    snapshot_path.write_text(json.dumps(snapshot, indent=2) + '\n')
    estimator = None
    previous_name = None
    try:
        for case in config.cases:
            if case.depth_est_name != previous_name:
                estimator = None
                estimator = create_depth_estimator(case.depth_est_name)
                previous_name = case.depth_est_name
            csv_path = output_dir / f'{case.depth_est_name}.csv'
            # CSV alone is sufficient only when artifacts were explicitly disabled.
            if not config.record_rollouts and config.resume and evaluation_done(
                    csv_path, case.task_name, case.depth_est_name,
                    case.camera_names, case.resolution, case.mask_input):
                continue
            model.set_num_demos(config.num_demos)
            run_evaluation(case.task_name, model, estimator, case.depth_est_name,
                           num_demos=config.num_demos, num_demo_iter=config.num_demo_iter,
                           num_rollouts=config.num_rollouts, headless=config.headless,
                           camera_names=list(case.camera_names), resolution=case.resolution,
                           mask_input=case.mask_input, load_cached_demo=config.load_cached_demo,
                           use_gt=case.depth_est_name == 'gt', debug=config.debug,
                           record_rollouts=config.record_rollouts, rollout_output_dir=str(output_dir),
                           video_fps=config.video_fps, resume=config.resume,
                           protocol=config.protocol, demo_seed=config.demo_seed,
                           rollout_seed=config.rollout_seed, demo_cache_dir=config.demo_cache_dir,
                           experiment_id=run_id, checkpoint_path=args.checkpoint_path)
    finally:
        del estimator
        del model
    return snapshot


if __name__ == '__main__':
    main()
