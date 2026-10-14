import os
import pdb
import cv2
import argparse
import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as Rot
import zarr
from tqdm import tqdm
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaIK, EndEffectorPoseViaPlanning
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment, TaskEnvironment
from rlbench.observation_config import CameraConfig, ObservationConfig
from rlbench.tasks import PhoneOnBase
from tqdm import trange
from rlbench.tasks import *
from rlb_cameras import RLBCameras
from pcd_scaler import scale_icp
from utils import visualise_pcds
import matplotlib.pyplot as plt
from termcolor import cprint
from rlb_point_cloud import farthest_point_sampling, preprocess_point_cloud
import gc
from utils import pose_7dof_to_6dof, relative_transform, pose_7dof_to_transform, transform_pcd


# Some examples of RLBench tasks
TASK_NAMES = {
    'lift_lid': TakeLidOffSaucepan,
    'phone_on_base': PhoneOnBase,
    'open_box': OpenBox,
    'slide_block': SlideBlockToTarget,
    'close_box': CloseBox,
    'basketball': BasketballInHoop,
    'buzz': BeatTheBuzz,
    'close_microwave': CloseMicrowave,
    'plate_out': TakePlateOffColoredDishRack,
    'toilet_seat_down': ToiletSeatDown,
    'toilet_seat_up': ToiletSeatUp,
    'toilet_roll_off': TakeToiletRollOffStand,
    'open_microwave': OpenMicrowave,
    'lamp_on': LampOn,
    'umbrella_out': TakeUmbrellaOutOfUmbrellaStand,
    'push_button': PushButton,
    'put_rubbish': PutRubbishInBin,
    'close_jar': CloseJar,
    'open_jar': OpenJar,
    'meat_off_grill': MeatOffGrill
}

class OnlineZarrWriter:
    """
    Stream steps to a Zarr store without buffering everything in memory.
    Axis 0 is resizable. Keeps an 'episode_ends' vector in /meta.
    """
    def __init__(
        self,
        store_path: str,
        point_cloud_shape=(1024, 6),  # (points, xyz+rgb)
        state_dim=7,
        action_dim=7,
        img_shape=(2, 256, 256, 3),
        depth_shape=(2, 256, 256),
        mask_shape=(2, 256, 256),
        cam_extrinsics_shape=(2, 4, 4),
        cam_intrinsics_shape=(2, 3, 3),
        chunk_steps=1000,
        overwrite=False,
    ):
        comp = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)
        mode = "w" if overwrite else "a"

        self.root = zarr.open_group(store_path, mode=mode)
        self.data = self.root.require_group("data")
        self.meta = self.root.require_group("meta")

        # Resizable datasets on axis 0
        self.img = self._req(
            self.data, "img",
            shape=(0, *img_shape),
            chunks=(chunk_steps, *img_shape),
            dtype="uint8",
            compressor=comp,
        )
        self.point_cloud = self._req(
            self.data, "point_cloud",
            shape=(0, *point_cloud_shape),
            chunks=(chunk_steps, *point_cloud_shape),
            dtype="float32",
            compressor=comp,
        )
        self.depth = self._req(
            self.data, "depth",
            shape=(0, *depth_shape),
            chunks=(chunk_steps, *depth_shape),
            dtype="float32",
            compressor=comp,
        )
        self.mask = self._req(
            self.data, "mask",
            shape=(0, *mask_shape),
            chunks=(chunk_steps, *mask_shape),
            dtype="uint8",
            compressor=comp,
        )
        self.cam_extrinsics = self._req(
            self.data, "cam_extrinsics",
            shape=(0, *cam_extrinsics_shape),
            chunks=(chunk_steps, *cam_extrinsics_shape),
            dtype="float32",
            compressor=comp,
        )
        self.cam_intrinsics = self._req(
            self.meta, "cam_intrinsics",
            shape=cam_intrinsics_shape,
            chunks=cam_intrinsics_shape,
            dtype="float32",
            compressor=comp,
        )
        self.action = self._req(
            self.data, "action",
            shape=(0, action_dim),
            chunks=(chunk_steps, action_dim),
            dtype="float32",
            compressor=comp,
        )
        self.state = self._req(
            self.data, "state",
            shape=(0, state_dim),
            chunks=(chunk_steps, state_dim),
            dtype="float32",
            compressor=comp,
        )
        self.episode_ends = self._req(
            self.meta, "episode_ends",
            shape=(0,), chunks=(max(100, chunk_steps),),
            dtype="int64", compressor=comp,
        )

        self._episode_start = None
        self._episode_meta_start = None
        self._recover_incomplete_episode()

    def _req(self, group, name, **kwargs):
        if name in group:
            return group[name]
        return group.require_dataset(name=name, **kwargs)

    def _step_arrays(self):
        return (
            self.img,
            self.point_cloud,
            self.depth,
            self.mask,
            self.cam_extrinsics,
            self.action,
            self.state,
        )

    def _recover_incomplete_episode(self):
        """Discard fields written after the last valid episode marker."""
        arrays = self._step_arrays()
        min_array_len = min(arr.shape[0] for arr in arrays)
        episode_ends = self.episode_ends[:]

        valid_episode_count = 0
        previous_end = 0
        for episode_end in episode_ends:
            episode_end = int(episode_end)
            if episode_end <= previous_end or episode_end > min_array_len:
                break
            previous_end = episode_end
            valid_episode_count += 1

        committed_steps = previous_end
        needs_recovery = (
            valid_episode_count != len(episode_ends)
            or any(arr.shape[0] != committed_steps for arr in arrays)
        )
        if needs_recovery:
            lengths = [arr.shape[0] for arr in arrays]
            cprint(
                "Recovering incomplete Zarr episode: "
                f"field_lengths={lengths}, committed_steps={committed_steps}",
                "yellow",
            )
            for arr in arrays:
                _grow_first_axis(arr, committed_steps)
            _grow_first_axis(self.episode_ends, valid_episode_count)

        self._n_steps = committed_steps
        self._n_eps = valid_episode_count

    def begin_episode(self):
        if self._episode_start is not None:
            raise RuntimeError("An episode transaction is already active.")
        self._episode_start = self._n_steps
        self._episode_meta_start = self._n_eps

    def abort_episode(self):
        """Roll back every field and marker written by the active episode."""
        if self._episode_start is None:
            return
        for arr in self._step_arrays():
            _grow_first_axis(arr, self._episode_start)
        _grow_first_axis(self.episode_ends, self._episode_meta_start)
        self._n_steps = self._episode_start
        self._n_eps = self._episode_meta_start
        self._episode_start = None
        self._episode_meta_start = None

    def set_cam_intrinsics_once(self, intrinsics: np.ndarray):
        """
        intrinsics: (Ncam,3,3) float
        Writes once (or overwrites) into /meta/cam_intrinsics.
        """
        intrinsics = intrinsics.astype(np.float32, copy=False)
        assert intrinsics.shape == self.cam_intrinsics.shape, \
            f"Expected intrinsics shape {self.cam_intrinsics.shape}, got {intrinsics.shape}"
        self.cam_intrinsics[:] = intrinsics

    @property
    def n_steps(self):
        return self._n_steps

    @property
    def n_episodes(self):
        return self._n_eps

    def append_step(self, point_cloud, state, action,
                    img, depth, mask, cam_extrinsics):
        """Append a single step (T=1)."""
        if self._episode_start is None:
            raise RuntimeError("Call begin_episode() before append_step().")

        # Ensure right shapes: (T,H,W,C) / (T,H,W) -> add batch axis
        if img.ndim == 4:
            img = img[None, ...]
        if depth.ndim == 3:
            depth = depth[None, ...]
        if mask.ndim == 3:
            mask = mask[None, ...]
        if cam_extrinsics.ndim == 3:
            cam_extrinsics = cam_extrinsics[None, ...]
        if point_cloud.ndim == 2:
            point_cloud = point_cloud[None, ...]
        if state.ndim == 1:
            state = state[None, ...]
        if action.ndim == 1:
            action = action[None, ...]

        start, end = self._n_steps, self._n_steps + 1

        _grow_first_axis(self.img, end);                self.img[start:end] = img
        _grow_first_axis(self.point_cloud, end);        self.point_cloud[start:end] = point_cloud
        _grow_first_axis(self.depth, end);              self.depth[start:end] = depth
        _grow_first_axis(self.mask, end);               self.mask[start:end] = mask
        _grow_first_axis(self.cam_extrinsics, end);     self.cam_extrinsics[start:end] = cam_extrinsics
        _grow_first_axis(self.action, end);             self.action[start:end] = action
        _grow_first_axis(self.state, end);              self.state[start:end] = state

        self._n_steps = end


    def end_episode(self):
        """Mark current step count as an episode end index."""
        # episode_ends is the commit boundary used to discard partial episodes
        # when reopening the store after a failed or interrupted collection.
        if self._episode_start is None:
            raise RuntimeError("No episode transaction is active.")
        if self._n_steps == self._episode_start:
            raise RuntimeError("Cannot commit an empty episode.")
        _grow_first_axis(self.episode_ends, self._n_eps + 1)
        self.episode_ends[self._n_eps] = self._n_steps
        self._n_eps += 1
        self._episode_start = None
        self._episode_meta_start = None


def _grow_first_axis(arr, new_len: int):
    """Compatible resize for zarr 2.x: pass full shape, no axis kw."""
    arr.resize((new_len,) + arr.shape[1:])


def initialise_rlbench(task_name='phone_on_base',
                       camera_names=['front', 'left_shoulder', 'right_shoulder', 'wrist'],
                       resolution=128,
                       headless=False,
                       restrict_rot=True):
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

        cam.set_resolution([resolution, resolution])
        cam_mask.set_resolution([resolution, resolution])

    return env, task


def init_zarr(save_data_path, num_points, action_dim=7, state_dim=7):
    root = zarr.open(save_data_path, mode="a")      # "a" = create or open
    z_data = root.require_group("data")
    z_meta = root.require_group("meta")

    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)

    # Create resizable arrays if missing
    if "point_cloud" not in z_data:
        z_data.create_dataset(
            "point_cloud",
            shape=(0, num_points, 6),          # start empty
            maxshape=(None, num_points, 6),    # append along axis 0
            chunks=(100, num_points, 6),
            dtype="float32",
            compressor=compressor,
        )
    if "img" not in z_data:
        z_data.create_dataset(
            "img",
            shape=(0, 2, 256, 256, 3),
            maxshape=(None, 2, 256, 256, 3),
            chunks=(100, 2, 256, 256, 3),
            dtype="uint8",
            compressor=compressor,
        )
    if "mask" not in z_data:
        z_data.create_dataset(
            "mask",
            shape=(0, 2, 256, 256),
            maxshape=(None, 2, 256, 256),
            chunks=(100, 2, 256, 256),
            dtype="uint8",
            compressor=compressor,
        )
    if "depth" not in z_data:
        z_data.create_dataset(
            "depth",
            shape=(0, 2, 256, 256),
            maxshape=(None, 2, 256, 256),
            chunks=(100, 2, 256, 256),
            dtype="float32",
            compressor=compressor,
        )
    if "cam_extrinsics" not in z_data:
        z_data.create_dataset(
            "cam_extrinsics",
            shape=(0, 2, 4, 4),
            maxshape=(None, 2, 4, 4),
            chunks=(100, 2, 4, 4),
            dtype="float32",
            compressor=compressor,
        )
    if "state" not in z_data:
        z_data.create_dataset(
            "state",
            shape=(0, state_dim),
            maxshape=(None, state_dim),
            chunks=(100, state_dim),
            dtype="float32",
            compressor=compressor,
        )
    if "action" not in z_data:
        z_data.create_dataset(
            "action",
            shape=(0, action_dim),
            maxshape=(None, action_dim),
            chunks=(100, action_dim),
            dtype="float32",
            compressor=compressor,
        )

    if "episode_ends" not in z_meta:
        z_meta.create_dataset(
            "episode_ends",
            shape=(0,),
            maxshape=(None,),
            chunks=(100,),
            dtype="int64",
            compressor=compressor,
        )

    return root  # you’ll use root["data"]["..."] etc.


def append_step(z_root, img, depth, mask, point_cloud, state, action):
    """Append one step (pcd/state/action) to zarr. Returns the new total count."""
    z_data = z_root["data"]

    # Current length (number of steps already stored)
    cur = z_data["state"].shape[0]

    # Ensure dtype/shape
    img = img.astype("uint8", copy=False)                       # (2, 256, 256, 3)
    depth = depth.astype("float32", copy=False)                 # (2, 256, 256)
    mask = mask.astype("uint8", copy=False)                     # (2, 256, 256)
    point_cloud = point_cloud.astype("float32", copy=False)     # (num_points, 6)
    state = state.astype("float32", copy=False)                 # (7,)
    action = action.astype("float32", copy=False)               # (7,)

    # Resize by +1 along axis 0, then fill the last row
    img = np.expand_dims(img, axis=0)                           # (1, 2, 256, 256, 3)
    depth = np.expand_dims(depth, axis=0)                       # (1, 2, 256, 256)
    mask = np.expand_dims(mask, axis=0)                         # (1, 2, 256, 256)
    point_cloud = np.expand_dims(point_cloud, axis=0)           # (1, num_points, 6)
    state = np.expand_dims(state, axis=0)                       # (1, 7)
    action = np.expand_dims(action, axis=0)                     # (1, 7)

    z_data["img"].append(img)
    z_data["depth"].append(depth)
    z_data["mask"].append(mask)
    z_data["point_cloud"].append(point_cloud)
    z_data["state"].append(state)
    z_data["action"].append(action)

    return cur + 1


def get_demos(
        task: TaskEnvironment,
        task_name: str,
        num_demos,
        depth_estimator=None,
        num_points: int = 1024,
        camera_names=['front', 'left_shoulder'],
        mask_input: bool = False,
        resolution: int = 128,
        use_gt: bool = True,
        stride: int = 1,
        debug: bool = False,
        overwrite: bool = False,
    ):
    def write_intrinsics_from_task(task, camera_names, writer):
        intrinsics = []
        for name in camera_names:
            if name == 'front':
                camera = task._scene._cam_front
                K = camera.get_intrinsic_matrix()
            elif name == 'left_shoulder':
                camera = task._scene._cam_over_shoulder_left
                K = camera.get_intrinsic_matrix()
            elif name == 'right_shoulder':
                camera = task._scene._cam_over_shoulder_right
                K = camera.get_intrinsic_matrix()
            elif name == 'wrist':
                camera = task._scene._cam_wrist
                K = camera.get_intrinsic_matrix()
            else:
                raise ValueError(f"Unknown camera name: {name}")
            intrinsics.append(K)
        intrinsics = np.stack(intrinsics, axis=0)  # (Ncam,3,3)
        writer.set_cam_intrinsics_once(intrinsics)

    # Open appendable Zarr once
    save_path = f'3D-Diffusion-Policy/3D-Diffusion-Policy/data/rlb_{task_name}.zarr'
    writer = OnlineZarrWriter(
        store_path=save_path,
        point_cloud_shape=(1024, 3),
        state_dim=7,
        action_dim=7,
        img_shape=(2, 256, 256, 3),
        depth_shape=(2, 256, 256),
        mask_shape=(2, 256, 256),
        cam_extrinsics_shape=(2, 4, 4),
        cam_intrinsics_shape=(len(camera_names), 3, 3),
        chunk_steps=100,
        overwrite=overwrite,
    )
    write_intrinsics_from_task(task, camera_names, writer)

    if writer.n_episodes:
        cprint(
            f"Resuming {task_name} from episode {writer.n_episodes}/{num_demos}",
            "yellow",
        )

    for i in tqdm(
        range(writer.n_episodes, num_demos),
        desc='Collecting demos',
        total=num_demos,
        initial=writer.n_episodes,
        leave=False,
    ):
        done = False
        while not done:
            try:
                demos = task.get_demos(1, live_demos=True, max_attempts=1000, random_eef_start=True)
                done = True
            except Exception:
                continue

            try:
                process_demo_streaming(
                    task=task,
                    demo=demos[0],
                    depth_estimator=depth_estimator,
                    num_points=num_points,
                    camera_names=camera_names,
                    mask_input=mask_input,
                    resolution=resolution,
                    use_gt=use_gt,
                    debug=debug,
                    writer=writer,
                    stride=stride,
                )
            finally:
                del demos
                gc.collect()
            print(f'Collected demo {i+1}/{num_demos}, total_steps: {writer.n_steps}')

    # Optional: consolidate metadata for faster future opens
    try:
        zarr.consolidate_metadata(writer.root.store)
    except Exception:
        pass

    cprint(f"Saved streaming Zarr at {save_path}", "green")
    cprint(f"steps={writer.n_steps}, episodes={writer.n_episodes}", "green")


def process_demo_streaming(task,
                           demo,
                           depth_estimator,
                           num_points: int = 1024,
                           camera_names: list = ['front', 'left_shoulder', 'right_shoulder', 'wrist'],
                           mask_input: bool = False,
                           resolution: int = 128,
                           use_gt: bool = False,
                           debug: bool = False,
                           writer: 'OnlineZarrWriter' = None,
                           stride: int = 1,
                           include_last: bool = True):

    assert writer is not None, "writer must be provided"
    assert stride >= 1, "stride must be >= 1"

    episode_length = len(demo)

    # Indices we will keep
    keep_idxs = list(range(0, episode_length, stride))
    if include_last and keep_idxs[-1] != episode_length - 1:
        keep_idxs.append(episode_length - 1)

    writer.begin_episode()
    try:
        for k, step_idx in enumerate(keep_idxs):
            obs = demo[step_idx]

            # Build point cloud
            obs_pointcloud, obs_imgs, obs_depths, obs_masks, obs_extrinsics, obs_intrinsics = get_point_cloud(
                task, obs, depth_estimator,
                camera_names=camera_names, mask_input=mask_input,
                resolution=resolution, use_gt=use_gt, debug=debug
            )
            obs_pointcloud = preprocess_point_cloud(obs_pointcloud, use_cuda=True)  # -> (1024, 3)

            # Transform to EEF
            T_WE = pose_7dof_to_transform(obs.gripper_pose)
            obs_pointcloud = transform_pcd(obs_pointcloud, np.linalg.inv(T_WE))

            # Robot state (7)
            robot_state = pose_7dof_to_6dof(obs.gripper_pose)
            gripper_state = obs.gripper_open
            robot_state = np.hstack((robot_state, gripper_state)).astype(np.float32)

            # Action (7)
            if k < len(keep_idxs) - 1:
                next_idx = keep_idxs[k + 1]
                next_obs = demo[next_idx]
                next_state = pose_7dof_to_6dof(next_obs.gripper_pose)
                next_gripper = next_obs.gripper_open
                rel_action = relative_transform(robot_state[:6], next_state)
                action = np.hstack((rel_action, next_gripper)).astype(np.float32)
            else:
                action = np.zeros((7,), dtype=np.float32)

            # Append single step to Zarr (img/depth dummies kept for schema compatibility)
            writer.append_step(
                point_cloud=obs_pointcloud.astype(np.float32),
                state=robot_state.astype(np.float32),
                action=action.astype(np.float32),
                img=obs_imgs,
                depth=obs_depths,
                mask=obs_masks,
                cam_extrinsics=obs_extrinsics
            )

        # Mark end-of-episode only after every field has been written.
        writer.end_episode()
    except BaseException:
        writer.abort_episode()
        raise


def run_estimator(name, img, intrinsics, extrinsics):
    print(f'Running depth estimator: {name}')
    if name == 'gt':
        depth_estimator = None
    elif name == 'vggt':
        from depth_estimators.vggt_est import VGGT
        depth_estimator = VGGT()
    elif name == 'map_anything':
        from depth_estimators.map_anything_est import MapAnythingEst
        depth_estimator = MapAnythingEst()
    elif name == 'mast3r':
        from depth_estimators.mast3r_est import Mast3rEst
        depth_estimator = Mast3rEst()
    elif name == 'moge':
        from depth_estimators.moge_depth import MoGeDepth
        depth_estimator = MoGeDepth()
    elif name == 'pi3':
        from depth_estimators.pi3_est import Pi3Est
        depth_estimator = Pi3Est()

    est_data = depth_estimator.infer(img, intrinsics, extrinsics)

    return est_data


def get_point_cloud_all(task,
                    obs,
                    depth_est_names,
                    previous_scale=None,
                    camera_names=['front', 'left_shoulder', 'right_shoulder', 'wrist'],
                    mask_input: bool = False,
                    resolution: int = 128,
                    use_gt: bool = False,
                    debug: bool = False):
    depth_estimator = None
    pcd_list = []
    for est_name in depth_est_names:
        if est_name == 'gt':
            use_gt = True
        else:
            use_gt = False

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

            intrinsics.append(camera.get_intrinsic_matrix())
            extrinsics.append(camera.get_matrix() if name != 'wrist' else obs.wrist_matrix)

        # Visualise rgbs and masked rgbs
        if mask_input:
            masked_rgbs = []
            for i in range(len(rgbs)):
                masked_rgb = rgbs[i].copy()
                masked_rgb[~masks[i]] = 0
                masked_rgbs.append(masked_rgb)

        if not use_gt:
            if mask_input:
                est_data = run_estimator(est_name, masked_rgbs, intrinsics, extrinsics)
            else:
                est_data = run_estimator(est_name, rgbs, intrinsics, extrinsics)

            if 'depth' in est_data.keys():
                depths = [data for data in est_data['depth']]
            elif 'pts3d' in est_data.keys():
                pcd = est_data['pts3d']
                visualise_pcds([pcd], extrinsics, intrinsics, gt_pcds)

        # Mask erosion to remove edges
        for i, mask in enumerate(masks):
            if resolution == 512:
                kernel = np.ones((5, 5), np.uint8)  # PhoneOnBase
            elif resolution == 256:
                kernel = np.ones((3, 3), np.uint8)
            else:
                kernel = np.ones((2, 2), np.uint8)
            masks[i] = cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)

        repr = RLBCameras.reproject_to_pcd
        if len(depths) > len(intrinsics):
            depths = depths[:len(intrinsics)]
        reprojected_pcds = [repr(depths[i], intrinsics[i], extrinsics[i], masks[i]) for i in range(len(depths))]

        if debug:
            visualise_pcds(reprojected_pcds, extrinsics, intrinsics, gt_pcds)

        # Outlier removal for specific depth estimators
        if name == 'map_anything':
            for i, pcd in enumerate(reprojected_pcds):
                pcd_noisy = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd))
                nb_neighbors = int(70 * resolution ** 2 / 700 ** 2)
                pcd_sor, ind_sor = pcd_noisy.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=2.0)
                reprojected_pcds[i] = np.asarray(pcd_sor.points)

        if use_gt or len(reprojected_pcds) == 1:
            scaled_pcds = reprojected_pcds

            pcd_list.append(np.concatenate(scaled_pcds, axis=0))
            continue
        else:
            try:
                scaled_pcds, best_scale = scale_icp(reprojected_pcds,
                                                    extrinsics,
                                                    previous_scale=None,
                                                    n_initialisations=5,
                                                    max_iterations=100,
                                                    intrinsics=(task._scene._cam_front.get_resolution(), intrinsics[0]),
                                                    visualise=False)
            except:
                pdb.set_trace()

        # Visualise unscaled pcds
        if debug:
            visualise_pcds(scaled_pcds, extrinsics, intrinsics, gt_pcds)

        pruned_pcds = []
        for scale_idx in range(len(scaled_pcds)):
            if scaled_pcds[scale_idx].shape[0] != 0:
                pruned_pcds.append(scaled_pcds[scale_idx])

        pcd_list.append(np.concatenate(pruned_pcds, axis=0))

    return pcd_list


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
            rgbs.append(getattr(obs, f'{name}_rgb'))
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

        intrinsics.append(camera.get_intrinsic_matrix())
        extrinsics.append(camera.get_matrix() if name != 'wrist' else obs.wrist_matrix)

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
        if resolution == 512:
            kernel = np.ones((5, 5), np.uint8)  # PhoneOnBase
        elif resolution == 256:
            kernel = np.ones((3, 3), np.uint8)
        else:
            kernel = np.ones((2, 2), np.uint8)
        masks[i] = cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)

    repr = RLBCameras.reproject_to_pcd
    if len(depths) > len(intrinsics):
        depths = depths[:len(intrinsics)]
    reprojected_pcds = [repr(depths[i], intrinsics[i], extrinsics[i], masks[i]) for i in range(len(depths))]

    if debug:
        visualise_pcds(reprojected_pcds, extrinsics, intrinsics, gt_pcds)

    # Outlier removal for specific depth estimators
    if depth_estimator is not None and depth_estimator.name == 'map_anything':
        for i, pcd in enumerate(reprojected_pcds):
            pcd_noisy = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd))
            nb_neighbors = int(70 * resolution ** 2 / 700 ** 2)
            pcd_sor, ind_sor = pcd_noisy.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=2.0)
            reprojected_pcds[i] = np.asarray(pcd_sor.points)

    # Output observations
    obs_imgs = np.stack(rgbs, axis=0)
    obs_depths = np.stack(depths, axis=0)
    obs_masks = np.stack(masks, axis=0)
    obs_extrinsics = np.stack(extrinsics, axis=0)
    obs_intrinsics = np.stack(intrinsics, axis=0)

    if use_gt or len(reprojected_pcds) == 1:
        scaled_pcds = reprojected_pcds
        return np.concatenate(scaled_pcds, axis=0), obs_imgs, obs_depths, obs_masks, obs_extrinsics, obs_intrinsics
    else:
        try:
            scaled_pcds, best_scale = scale_icp(reprojected_pcds,
                                                extrinsics,
                                                previous_scale=None,
                                                n_initialisations=5,
                                                max_iterations=100,
                                                intrinsics=(task._scene._cam_front.get_resolution(), intrinsics[0]),
                                                visualise=False)
        except:
            pdb.set_trace()

    # Visualise unscaled pcds
    if debug:
        visualise_pcds(scaled_pcds, extrinsics, intrinsics, gt_pcds)

    pruned_pcds = []
    for scale_idx in range(len(scaled_pcds)):
        if scaled_pcds[scale_idx].shape[0] != 0:
            pruned_pcds.append(scaled_pcds[scale_idx])

    return np.concatenate(pruned_pcds, axis=0), obs_imgs, obs_depths, obs_masks, obs_extrinsics, obs_intrinsics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='discard an existing task Zarr store instead of resuming it',
    )
    args = parser.parse_args()
    headless = args.headless
    debug = args.debug
    stride = args.stride
    overwrite = args.overwrite

    num_demos = 100
    num_demo_iter = 20
    num_rollouts = 10

    ####################################################################################################################

    # Task selection and demo count are configured here; the CLI only controls
    # headless/debug mode, stride, and whether an existing store is overwritten.
    task_names = [
        # 'phone_on_base',
        # 'open_box',
        'close_box',
        'plate_out',
        'basketball',
        # 'close_microwave',
        # 'toilet_seat_down',
        # 'toilet_seat_up',
        # 'toilet_roll_off',
        # 'meat_off_grill'
    ]

    depth_est_names = [
        'gt',
        # 'vggt',
        # 'map_anything',
        # 'mast3r',
        # 'moge',
        # 'pi3'
    ]

    depth_estimators = []
    for name in depth_est_names:
        if name == 'gt':
            depth_estimators.append(None)
        elif name == 'vggt':
            from depth_estimators.vggt_est import VGGT
            depth_estimators.append(VGGT())
        elif name == 'map_anything':
            from depth_estimators.map_anything_est import MapAnythingEst
            depth_estimators.append(MapAnythingEst())
        elif name == 'mast3r':
            from depth_estimators.mast3r_est import Mast3rEst
            depth_estimators.append(Mast3rEst())
        elif name == 'moge':
            from depth_estimators.moge_depth import MoGeDepth
            depth_estimators.append(MoGeDepth())
        elif name == 'pi3':
            from depth_estimators.pi3_est import Pi3Est
            depth_estimators.append(Pi3Est())

    camera_setups = [
        ['front'],
        ['front', 'left_shoulder'],
        ['front', 'left_shoulder', 'wrist']
    ]

    resolutions = [
        128,
        256,
        512
    ]

    ####################################################################################################################
    for task_name in task_names:
        env, task = initialise_rlbench(task_name=task_name,
                                        camera_names=camera_setups[1],
                                        resolution=resolutions[1],
                                        headless=headless,
                                        restrict_rot=True)

        get_demos(
            task,
            task_name=task_name,
            num_demos=num_demos,
            depth_estimator=depth_estimators[0],
            num_points=1024,
            camera_names=camera_setups[1],
            mask_input=False,
            resolution=resolutions[1],
            use_gt=True,
            stride=stride,
            debug=debug,
            overwrite=overwrite,
        )

        env.shutdown()
