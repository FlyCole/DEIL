from typing import Tuple

import open3d as o3d
from scipy.spatial.transform import Rotation as Rot
import numpy as np
import matplotlib.pyplot as plt

# RLBench camera axes are left/up/forward; OpenCV uses right/down/forward.
# For camera-to-world poses, apply this basis change on the right: T_WC @ flip.
RLBench_TO_OPENCV_CAMERA = np.array(
    [[-1, 0, 0, 0],
     [0, -1, 0, 0],
     [0, 0, 1, 0],
     [0, 0, 0, 1]],
    dtype=np.float32,
)

def Tmtx_from_posevec(t: np.ndarray,
                      q: np.ndarray = None,
                      rvec: np.ndarray = None) -> np.ndarray:
    """
    Convert a pose vector to a transformation matrix.

    Args:
        t (np.ndarray): Translation (3,)
        q (np.ndarray, optional): Quaternion (4,), if provided will be used instead of rvec.
        rvec (np.ndarray, optional): Rotation vector (3,), if provided will be used instead of q.

    Returns:
        np.ndarray: A 4x4 transformation matrix.
    """
    if q is not None:
        if q.shape != (4,):
            raise ValueError("Quaternion must be of shape (4,).")
        r = Rot.from_quat(q)
    elif rvec is not None:
        if rvec.shape != (3,):
            raise ValueError("Rotation vector must be of shape (3,).")
        r = Rot.from_rotvec(rvec)
    else:
        raise ValueError("Either quaternion (q) or rotation vector (rvec) must be provided.")
    if t.shape != (3,):
        raise ValueError("Translation vector must be of shape (3,).")
    T = np.eye(4)
    T[:3, :3] = r.as_matrix()
    T[:3, 3] = t
    return T
    

def intrinsics_to_o3d_format(intrinsics_matrix: np.ndarray,
                             image_shape: Tuple[int, int]) -> o3d.camera.PinholeCameraIntrinsic:
    """
    Convert a camera intrinsics matrix to Open3D format.

    Args:
        intrinsics_matrix (np.ndarray): A 3x3 camera intrinsics matrix.
        image_shape (Tuple[int, int]): The shape of the image as (height, width).

    Returns:
        o3d.camera.PinholeCameraIntrinsic: Open3D camera intrinsic object.
    """
    if intrinsics_matrix.shape != (3, 3):
        raise ValueError("Intrinsics matrix must be of shape (3, 3).")

    fx = intrinsics_matrix[0, 0]
    fy = intrinsics_matrix[1, 1]
    cx = intrinsics_matrix[0, 2]
    cy = intrinsics_matrix[1, 2]

    return o3d.camera.PinholeCameraIntrinsic(
        width=image_shape[1],
        height=image_shape[0],
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy
    )
    

def transform_pcd(pcd, T):
    return np.matmul(T[:3, :3], pcd.T).T + T[:3, 3]


def transform_to_7d_pose(T):
    """
    Convert a 4x4 transformation matrix to a 7-DOF pose.
    :param T: 4x4 transformation matrix.
    :return: 7dof pose [x, y, z, qx, qy, qz, qw].
    """
    rot = Rot.from_matrix(T[:3, :3])
    return np.concatenate([T[:3, 3], rot.as_quat()])


def transform_to_6d_pose(T):
    """
    Convert a 4x4 transformation matrix to a 6-DOF pose.
    :param T: 4x4 transformation matrix.
    :return: 6dof rotvec and translation.
    """
    rotvec = Rot.from_matrix(T[:3, :3]).as_rotvec()
    return np.concatenate([T[:3, 3], rotvec])


def pose_7dof_to_transform(pose):
    """
    Convert a pose to a 4x4 transformation matrix.
    :param pose: [x, y, z, qx, qy, qz, qw]
    :return: 4x4 transformation matrix.
    """
    T = np.eye(4)
    T[:3, 3] = pose[:3]
    T[:3, :3] = Rot.from_quat(pose[3:]).as_matrix()
    return T

def pose_6dof_to_transform(pose):
    """
    Convert a 6-DOF pose to a 4x4 transformation matrix.
    :param pose: [x, y, z, rx, ry, rz]
    :return: 4x4 transformation matrix.
    """
    # rx/ry/rz form an axis-angle rotation vector in radians, not Euler angles.
    T = np.eye(4)
    T[:3, 3] = pose[:3]
    T[:3, :3] = Rot.from_rotvec(pose[3:]).as_matrix()
    return T


def quat_pose_mul(pose1, pose2):
    """
    Multiply two 7-DOF poses.
    :param pose1: [x, y, z, qx, qy, qz, qw]
    :param pose2: [x, y, z, qx, qy, qz, qw]
    :return: result of multiplication as 7dof pose.
    """
    T1 = pose_7dof_to_transform(pose1)
    T2 = pose_7dof_to_transform(pose2)
    T_res = T1 @ T2
    return transform_to_7d_pose(T_res)

def rotvec_pose_mul(pose1, pose2):
    """
    Multiply two 6-DOF poses.
    :param pose1: [x, y, z, rx, ry, rz]
    :param pose2: [x, y, z, rx, ry, rz]
    :return: result of multiplication as 6dof pose.
    """
    T1 = pose_6dof_to_transform(pose1)
    T2 = pose_6dof_to_transform(pose2)
    T_res = T1 @ T2
    return transform_to_6d_pose(T_res)


def pose_7dof_to_6dof(pose):
    """
    Convert a 7-DOF pose to a 6-DOF pose.
    :param pose: [x, y, z, qx, qy, qz, qw]
    :return: 6dof rotvec and translation.
    """
    T = pose_7dof_to_transform(pose)
    rotvec = Rot.from_matrix(T[:3, :3]).as_rotvec()
    return np.concatenate([T[:3, 3], rotvec])


def relative_transform(pose_0, pose_1):
    if pose_0.shape[0] == pose_1.shape[0] == 7:
        T_0 = pose_7dof_to_transform(pose_0)
        T_1 = pose_7dof_to_transform(pose_1)
        T_rel = np.linalg.inv(T_0) @ T_1
        return transform_to_7d_pose(T_rel)
    elif pose_0.shape[0] == pose_1.shape[0] == 6:
        T_0 = pose_6dof_to_transform(pose_0)
        T_1 = pose_6dof_to_transform(pose_1)
        T_rel = np.linalg.inv(T_0) @ T_1
        return transform_to_6d_pose(T_rel)
    else:
        T_0 = pose_0
        T_1 = pose_1
        T_rel = np.linalg.inv(T_0) @ T_1

    return T_rel


def transform_pcd(pcd, T):
    return np.matmul(T[:3, :3], pcd.T).T + T[:3, 3]


def pose_error(T1, T2, rot_scale=0.01):
    dist_trans = np.linalg.norm(T1[:3, 3] - T2[:3, 3])
    dist_rot = Rot.from_matrix(T1[:3, :3] @ T2[:3, :3].T).magnitude()
    return dist_trans + rot_scale * dist_rot


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


def remove_statistical_outliers(point_cloud, nb_neighbors=20, std_ratio=2.0):
    # Create a PointCloud object from the NumPy array
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)

    # Perform statistical outlier removal
    [filtered_pcd, inlier_indices] = pcd.remove_statistical_outlier(nb_neighbors, std_ratio)

    # Convert the filtered PointCloud back to a NumPy array
    filtered_point_cloud = np.asarray(filtered_pcd.points)

    return filtered_point_cloud, inlier_indices


def _make_o3d_point_cloud(points, color=None, voxel_size=0.03):
    raw_points = np.asarray(points)
    if raw_points.size == 0:
        return None
    points = raw_points.reshape(-1, raw_points.shape[-1])
    valid = np.isfinite(points[:, :3]).all(axis=1)
    points = points[valid]
    if points.shape[0] == 0:
        return None

    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(points[:, :3])
    if points.shape[1] >= 6:
        rgb = points[:, 3:6].astype(np.float64)
        if rgb.size and rgb.max() > 1.0:
            rgb = rgb / 255.0
        rgb = np.clip(rgb, 0.0, 1.0)
        pcd_o3d.colors = o3d.utility.Vector3dVector(rgb)
    else:
        if color is None:
            color = [0.6, 0.6, 0.6]
        pcd_o3d.paint_uniform_color(color)
    if voxel_size is not None and voxel_size > 0:
        pcd_o3d = pcd_o3d.voxel_down_sample(voxel_size=voxel_size)
    return pcd_o3d


def _rlbench_camera_visualization(T_WC, K, scale=0.1):
    K_o3d = np.asarray(K, dtype=np.float64).copy()
    K_o3d[0, 0] = abs(K_o3d[0, 0])
    K_o3d[1, 1] = abs(K_o3d[1, 1])
    width = max(1, int(round(K_o3d[0, 2] * 2)))
    height = max(1, int(round(K_o3d[1, 2] * 2)))

    T_WC_cv = np.asarray(T_WC, dtype=np.float64) @ RLBench_TO_OPENCV_CAMERA
    return o3d.geometry.LineSet.create_camera_visualization(
        view_width_px=width,
        view_height_px=height,
        intrinsic=K_o3d,
        extrinsic=np.linalg.inv(T_WC_cv),
        scale=scale,
    )


def visualise_pcds(pcds, extrinsics, intrinsics, gt_pcd,
                   out_dir=None, iter_idx=0, suffix='', show=True,
                   show_gt=True, show_cameras=True, show_world_frame=True,
                   voxel_size=0.03, window_name='Point clouds'):
    palette = plt.get_cmap("tab10")
    pcd_list = []
    if pcds is None:
        pcds = []
    if isinstance(pcds, np.ndarray):
        pcds = [pcds]
    if gt_pcd is not None and isinstance(gt_pcd, np.ndarray):
        gt_pcd = [gt_pcd]

    for i, pcd in enumerate(pcds):
        pcd_o3d = _make_o3d_point_cloud(pcd, list(palette(i)[:3]), voxel_size)
        if pcd_o3d is not None:
            pcd_list.append(pcd_o3d)

    if show_gt and gt_pcd is not None:
        for i, pcd in enumerate(gt_pcd):
            pcd_o3d_gt = _make_o3d_point_cloud(pcd, [0.15, 0.75, 0.25], voxel_size)
            if pcd_o3d_gt is not None:
                pcd_list.append(pcd_o3d_gt)

    if show_world_frame:
        pcd_list.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2))

    if show_cameras and extrinsics is not None and intrinsics is not None:
        for i, T_WC in enumerate(extrinsics):
            K = intrinsics[min(i, len(intrinsics) - 1)]
            pcd_list.append(_rlbench_camera_visualization(T_WC, K, scale=0.1))

    if show and pcd_list:
        o3d.visualization.draw_geometries(pcd_list, window_name=window_name)

    if out_dir is None:
        return

    # Visualise using transparent background
    from open3d.visualization import rendering
    import os
    os.makedirs(out_dir, exist_ok=True)
    try:
        renderer = rendering.OffscreenRenderer(1000, 1000)
    except Exception as e:
        print(f"[WARN] Open3D offscreen renderer failed: {e}")
        return
    scene = renderer.scene
    scene.set_background([1, 1, 1, 0])

    # Basic lighting that looks good for papers
    scene.scene.set_sun_light([0.5, 0.5, -1.0], [1.0, 1.0, 1.0], 75000)
    scene.scene.enable_sun_light(True)
    scene.scene.set_indirect_light_intensity(1500)

    # Materials
    mat_pcd = rendering.MaterialRecord()
    mat_pcd.shader = "defaultUnlit"
    # mat_pcd.point_size = float(2.5)

    mat_line = rendering.MaterialRecord()
    mat_line.shader = "unlitLine"
    # mat_line.line_width = float(line_width)

    # Add geometries
    for idx, g in enumerate(pcd_list):
        name = f"g{idx}"
        if isinstance(g, o3d.geometry.PointCloud):
            scene.add_geometry(name, g, mat_pcd)
        elif isinstance(g, o3d.geometry.LineSet):
            scene.add_geometry(name, g, mat_line)
        else:
            # Mesh (coordinate frame)
            mat_mesh = rendering.MaterialRecord()
            mat_mesh.shader = "defaultLit"
            scene.add_geometry(name, g, mat_mesh)

    # Camera: auto-fit if not specified
    bounds = scene.bounding_box
    center = bounds.get_center()
    extent = np.linalg.norm(bounds.get_extent())
    lookat = center
    # eye = center + np.array([0.8, -1.2, 0.8]) * (0.8 * extent + 1e-6)
    eye = center + np.array([-1.0, -1.0, 2.0]) * (0.3 * extent + 1e-6)

    renderer.setup_camera(60.0, lookat, eye, (0, 0, 1))

    img = renderer.render_to_image()
    # Make background transparent
    img_data = np.asarray(img)
    # Add alpha channel
    alpha_channel = np.where(np.all(img_data[:, :, :3] <= 10, axis=-1), 0, 255).astype(np.uint8)
    img_data = np.dstack((img_data[:, :, :3], alpha_channel))
    img = o3d.geometry.Image(img_data)
    o3d.io.write_image(f"{out_dir}/{iter_idx}_pcd_{suffix}.png", img)

def pointcloud2_to_np(pc2_msg):
    """
    Assumes fields [x,y,z,r,g,b] as float32 each.
    Returns (N,6) np.float32
    """
    N = pc2_msg.width * pc2_msg.height
    arr = np.frombuffer(pc2_msg.data, dtype=np.float32).reshape(N, 6)
    return arr
