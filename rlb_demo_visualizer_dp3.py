import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
import zarr

from utils import pose_6dof_to_transform, transform_pcd


def clamp(value, low, high):
    return max(low, min(high, value))


def rotation_from_z(direction):
    direction = np.asarray(direction, dtype=np.float64)
    norm = np.linalg.norm(direction)
    if norm < 1e-9:
        return np.eye(3)

    target = direction / norm
    source = np.array([0.0, 0.0, 1.0])
    dot = float(np.clip(np.dot(source, target), -1.0, 1.0))

    if dot > 1.0 - 1e-9:
        return np.eye(3)
    if dot < -1.0 + 1e-9:
        return o3d.geometry.get_rotation_matrix_from_axis_angle(np.array([np.pi, 0.0, 0.0]))

    axis = np.cross(source, target)
    axis = axis / np.linalg.norm(axis)
    angle = np.arccos(dot)
    return o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)


def make_coordinate_frame(transform, size):
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    frame.transform(transform)
    return frame


def make_action_arrow(start, end, color, radius):
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    vec = end - start
    length = float(np.linalg.norm(vec))

    if length < 1e-5:
        marker = o3d.geometry.TriangleMesh.create_sphere(radius=max(radius * 2.0, 0.005))
        marker.translate(start)
        marker.paint_uniform_color(color)
        return marker

    cone_height = max(length * 0.22, radius * 4.0)
    cylinder_height = max(length - cone_height, radius * 2.0)
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=radius,
        cone_radius=radius * 2.4,
        cylinder_height=cylinder_height,
        cone_height=cone_height,
        resolution=16,
        cylinder_split=4,
        cone_split=1,
    )
    arrow.rotate(rotation_from_z(vec), center=(0.0, 0.0, 0.0))
    arrow.translate(start)
    arrow.paint_uniform_color(color)
    arrow.compute_vertex_normals()
    return arrow


def make_camera_frustum(intrinsic, extrinsic_cw, image_shape, scale):
    intrinsic = intrinsic.astype(np.float64, copy=True)
    intrinsic[0, 0] = abs(intrinsic[0, 0])
    intrinsic[1, 1] = abs(intrinsic[1, 1])

    height, width = image_shape
    frustum = o3d.geometry.LineSet.create_camera_visualization(
        view_width_px=int(width),
        view_height_px=int(height),
        intrinsic=intrinsic,
        extrinsic=np.linalg.inv(extrinsic_cw),
        scale=scale,
    )
    frustum.paint_uniform_color([1.0, 0.75, 0.1])
    return frustum


class RLBDemoVisualizer:
    def __init__(self, args):
        self.args = args
        self.root = zarr.open_group(args.zarr_path, mode="r")
        self.data = self.root["data"]
        self.meta = self.root["meta"]

        if args.pc_key not in self.data:
            keys = ", ".join(sorted(k for k in self.data.keys() if k.startswith("point_cloud")))
            raise KeyError(f"Missing /data/{args.pc_key}. Available point-cloud keys: {keys}")

        required = ["state", "action", "cam_extrinsics", args.pc_key]
        for key in required:
            if key not in self.data:
                raise KeyError(f"Missing /data/{key}")
        if "episode_ends" not in self.meta:
            raise KeyError("Missing /meta/episode_ends")
        if "cam_intrinsics" not in self.meta:
            raise KeyError("Missing /meta/cam_intrinsics")

        self.episode_ends = self.meta["episode_ends"][:].astype(np.int64)
        if len(self.episode_ends) == 0:
            raise ValueError("Dataset has no episodes")

        self.intrinsics = self.meta["cam_intrinsics"][:]
        self.num_steps = int(self.episode_ends[-1])
        self.num_episodes = len(self.episode_ends)
        self.image_shape = self._get_image_shape()

        if args.global_step is not None:
            self.global_step = clamp(args.global_step, 0, self.num_steps - 1)
            self.episode_idx = self._episode_for_global_step(self.global_step)
            self.local_step = self.global_step - self._episode_start(self.episode_idx)
        else:
            self.episode_idx = clamp(args.episode, 0, self.num_episodes - 1)
            self.local_step = clamp(args.step, 0, self._episode_len(self.episode_idx) - 1)
            self.global_step = self._episode_start(self.episode_idx) + self.local_step

        self.dynamic_geometries = []
        self.vis = o3d.visualization.VisualizerWithKeyCallback()

    def _get_image_shape(self):
        if "img" in self.data and len(self.data["img"].shape) >= 4:
            return int(self.data["img"].shape[2]), int(self.data["img"].shape[3])

        cx = abs(float(self.intrinsics[0, 0, 2]))
        cy = abs(float(self.intrinsics[0, 1, 2]))
        return max(1, int(round(cy * 2))), max(1, int(round(cx * 2)))

    def _episode_start(self, episode_idx):
        if episode_idx == 0:
            return 0
        return int(self.episode_ends[episode_idx - 1])

    def _episode_end(self, episode_idx):
        return int(self.episode_ends[episode_idx])

    def _episode_len(self, episode_idx):
        return self._episode_end(episode_idx) - self._episode_start(episode_idx)

    def _episode_for_global_step(self, global_step):
        return int(np.searchsorted(self.episode_ends, global_step, side="right"))

    def _set_global_step(self, global_step):
        self.global_step = clamp(global_step, 0, self.num_steps - 1)
        self.episode_idx = self._episode_for_global_step(self.global_step)
        self.local_step = self.global_step - self._episode_start(self.episode_idx)

    def _set_local_step(self, local_step):
        local_step = clamp(local_step, 0, self._episode_len(self.episode_idx) - 1)
        self.local_step = local_step
        self.global_step = self._episode_start(self.episode_idx) + local_step

    def _set_episode(self, episode_idx):
        self.episode_idx = clamp(episode_idx, 0, self.num_episodes - 1)
        self.local_step = 0
        self.global_step = self._episode_start(self.episode_idx)

    def _eef_transform(self, global_step):
        state = self.data["state"][global_step].astype(np.float64)
        return pose_6dof_to_transform(state[:6])

    def _point_cloud_world(self, global_step, eef_transform):
        pcd = self.data[self.args.pc_key][global_step]
        xyz = pcd[:, :3].astype(np.float64, copy=False)
        if self.args.pc_frame == "eef":
            xyz = transform_pcd(xyz, eef_transform)

        out = o3d.geometry.PointCloud()
        out.points = o3d.utility.Vector3dVector(xyz)

        if pcd.shape[1] >= 6:
            colors = pcd[:, 3:6].astype(np.float64, copy=False)
            if colors.max(initial=0.0) > 1.0:
                colors = colors / 255.0
            out.colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0))
        else:
            out.paint_uniform_color([0.68, 0.72, 0.78])

        if self.args.voxel_size > 0:
            out = out.voxel_down_sample(self.args.voxel_size)
        return out

    def _camera_geometries(self, global_step):
        extrinsics = self.data["cam_extrinsics"][global_step].astype(np.float64)
        geoms = []
        for cam_idx, extrinsic_cw in enumerate(extrinsics):
            geoms.append(make_coordinate_frame(extrinsic_cw, self.args.camera_frame_size))
            if self.args.show_frustums:
                intrinsic = self.intrinsics[min(cam_idx, len(self.intrinsics) - 1)]
                geoms.append(make_camera_frustum(
                    intrinsic=intrinsic,
                    extrinsic_cw=extrinsic_cw,
                    image_shape=self.image_shape,
                    scale=self.args.camera_frustum_scale,
                ))
        return geoms

    def _action_geometries(self):
        geoms = []
        ep_end = self._episode_end(self.episode_idx)
        palette = [
            [0.0, 0.8, 1.0],
            [0.0, 0.55, 1.0],
            [0.15, 0.35, 1.0],
            [0.45, 0.25, 1.0],
            [0.75, 0.2, 1.0],
        ]

        for offset in range(self.args.action_horizon):
            step = self.global_step + offset
            if step >= ep_end:
                break
            action = self.data["action"][step].astype(np.float64)
            T_we = self._eef_transform(step)
            T_delta = pose_6dof_to_transform(action[:6])
            T_next = T_we @ T_delta

            start = T_we[:3, 3]
            raw_end = T_next[:3, 3]
            end = start + (raw_end - start) * self.args.action_scale
            color = palette[min(offset, len(palette) - 1)]
            geoms.append(make_action_arrow(start, end, color, self.args.action_radius))

        return geoms

    def _make_geometries(self):
        T_we = self._eef_transform(self.global_step)
        geoms = [
            self._point_cloud_world(self.global_step, T_we),
            make_coordinate_frame(T_we, self.args.eef_frame_size),
        ]
        geoms.extend(self._camera_geometries(self.global_step))
        geoms.extend(self._action_geometries())
        return geoms

    def _print_status(self):
        ep_len = self._episode_len(self.episode_idx)
        print(
            f"episode {self.episode_idx + 1}/{self.num_episodes} "
            f"step {self.local_step}/{ep_len - 1} "
            f"global {self.global_step}/{self.num_steps - 1} "
            f"pc_key={self.args.pc_key}"
        )

    def _refresh(self, reset_bounding_box=False):
        for geom in self.dynamic_geometries:
            self.vis.remove_geometry(geom, reset_bounding_box=False)
        self.dynamic_geometries = self._make_geometries()
        for geom in self.dynamic_geometries:
            self.vis.add_geometry(geom, reset_bounding_box=reset_bounding_box)
        self._print_status()
        self.vis.poll_events()
        self.vis.update_renderer()

    def _register_key_callbacks(self):
        def refresh_after(fn):
            def callback(vis):
                fn()
                self._refresh(reset_bounding_box=False)
                return False
            return callback

        self.vis.register_key_callback(ord("N"), refresh_after(
            lambda: self._set_local_step(self.local_step + self.args.step_stride)))
        self.vis.register_key_callback(ord("B"), refresh_after(
            lambda: self._set_local_step(self.local_step - self.args.step_stride)))
        self.vis.register_key_callback(ord("F"), refresh_after(
            lambda: self._set_local_step(self.local_step + self.args.jump)))
        self.vis.register_key_callback(ord("R"), refresh_after(
            lambda: self._set_local_step(self.local_step - self.args.jump)))
        self.vis.register_key_callback(ord("E"), refresh_after(
            lambda: self._set_episode(self.episode_idx + 1)))
        self.vis.register_key_callback(ord("W"), refresh_after(
            lambda: self._set_episode(self.episode_idx - 1)))
        self.vis.register_key_callback(ord("G"), refresh_after(
            lambda: self._set_local_step(0)))
        self.vis.register_key_callback(ord("L"), refresh_after(
            lambda: self._set_local_step(self._episode_len(self.episode_idx) - 1)))
        self.vis.register_key_callback(ord("Q"), lambda vis: (vis.close(), False)[1])

    def run(self):
        title = f"RLBench demo: {Path(self.args.zarr_path).name}"
        self.vis.create_window(window_name=title, width=self.args.width, height=self.args.height)
        self.vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=self.args.world_frame_size))
        self._register_key_callbacks()
        render = self.vis.get_render_option()
        render.point_size = self.args.point_size
        render.background_color = np.asarray(self.args.background, dtype=np.float64)
        self._refresh(reset_bounding_box=True)
        self.vis.run()
        self.vis.destroy_window()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactively inspect collected RLBench DP3 Zarr demos in world coordinates."
    )
    parser.add_argument("--zarr_path", required=True, help="Path to an rlb_*.zarr dataset.")
    parser.add_argument("--pc_key", default="point_cloud",
                        help="Point cloud key to visualize, e.g. point_cloud_vggt.")
    parser.add_argument("--pc_frame", choices=["eef", "world"], default="eef",
                        help="Frame of the stored point cloud. Collected DP3 data uses eef.")
    parser.add_argument("--episode", type=int, default=0, help="Initial 0-based episode index.")
    parser.add_argument("--step", type=int, default=0, help="Initial 0-based step within episode.")
    parser.add_argument("--global_step", type=int, default=None, help="Initial global step index.")
    parser.add_argument("--step_stride", type=int, default=1, help="Steps moved by N/B.")
    parser.add_argument("--jump", type=int, default=10, help="Steps moved by F/R.")
    parser.add_argument("--action_horizon", type=int, default=3,
                        help="Number of future action arrows to draw from the current step.")
    parser.add_argument("--action_scale", type=float, default=1.0,
                        help="Scale applied to relative action translation arrows.")
    parser.add_argument("--action_radius", type=float, default=0.004)
    parser.add_argument("--point_size", type=float, default=3.0)
    parser.add_argument("--voxel_size", type=float, default=0.0,
                        help="Optional voxel downsample size for display only.")
    parser.add_argument("--world_frame_size", type=float, default=0.25)
    parser.add_argument("--eef_frame_size", type=float, default=0.08)
    parser.add_argument("--camera_frame_size", type=float, default=0.08)
    parser.add_argument("--camera_frustum_scale", type=float, default=0.12)
    parser.add_argument("--hide_frustums", dest="show_frustums", action="store_false")
    parser.set_defaults(show_frustums=True)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--background", type=float, nargs=3, default=[0.04, 0.04, 0.045])
    return parser.parse_args()


def print_controls():
    print("Controls:")
    print("  N/B: next/previous step")
    print("  F/R: jump forward/backward")
    print("  E/W: next/previous episode")
    print("  G/L: first/last step in episode")
    print("  Q: quit")


def main():
    args = parse_args()
    print_controls()
    viewer = RLBDemoVisualizer(args)
    viewer.run()


if __name__ == "__main__":
    main()
