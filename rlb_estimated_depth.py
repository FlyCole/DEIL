import pdb

import open3d as o3d
import numpy as np

from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaIK
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment, TaskEnvironment
from rlbench.observation_config import CameraConfig, ObservationConfig
from rlbench.demo import Demo

from rlbench.tasks import PhoneOnBase
from pcd_scaler import scale_icp
import os

import matplotlib.pyplot as plt
from rlb_cameras import RLBCameras
from pyrep import pyrep

from depth_estimators.moge_depth import MoGeDepth


pr = pyrep.PyRep()
obs_config = ObservationConfig()
obs_config.set_all(True)
action_mode = MoveArmThenGripper(
    arm_action_mode=EndEffectorPoseViaIK(),
    gripper_action_mode=Discrete()
)
env = Environment(action_mode,
                    './',
                    obs_config=obs_config,
                    headless=True)
env.launch()
task = env.get_task(PhoneOnBase)

cameras = RLBCameras(env._scene._cam_over_shoulder_left,
                     env._scene._cam_over_shoulder_right,
                     # env._scene._cam_front,
                     depth_estimator=MoGeDepth())

# cameras = RLBCameras(env._scene._cam_over_shoulder_left,
#                      env._scene._cam_over_shoulder_right,
                     # env._scene._cam_front)

cameras.change_resolution(700, 700)

# Get RGB images from the cameras
# rgb_images = cameras.get_rgbs()


for i in range(100):
    pr.step()


# # scene_pcd = cameras.get_gt_scene_pcds()
scene_pcd = cameras.get_gt_reprojected_pcd()
print(scene_pcd.shape)

# Ensure scene_pcd is a 2D numpy array with shape (N, 3)
scene_pcd = scene_pcd.reshape(-1, 3) #.astype(np.float64)
# Convert the scene point cloud to an Open3D point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(scene_pcd)

# scaled_pcds, gt_scales = cameras.get_scaled_pcd()
estimated_pcds = cameras.get_pcds_from_moge()

rlb2opencv = np.array([[-1, 0, 0, 0],
                       [0, -1, 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]])

extrinsics = cameras.get_extrinsics()
extrinsics = [rlb2opencv @ extrinsic for extrinsic in extrinsics]
for i in range(len(estimated_pcds)):
    pts_new = np.asarray(estimated_pcds[i].points)
    pts_new = np.concatenate((pts_new, np.ones(
        (pts_new.shape[0], 1), dtype=pts_new.dtype)), axis=1) @ (extrinsics[i]).T
    estimated_pcds[i].points = o3d.utility.Vector3dVector(pts_new[:, :3])


scaled_pcds, scales = scale_icp([np.asarray(pcd.points) for pcd in estimated_pcds],
                                cameras.get_extrinsics(), verbose=True)
pcd_list = []

vis = o3d.visualization.Visualizer()
vis.create_window()

for i in range(len(scales)):
    pts = np.asarray(estimated_pcds[i].points) * scales[i]
    pts -= np.dot(extrinsics[i][:3, 3], (scales[i] - 1))
    estimated_pcds[i].points = o3d.utility.Vector3dVector(pts)
    vis.add_geometry(estimated_pcds[i])

vis.run()

ext = cameras.get_extrinsics()
# # Create a list to hold the point clouds
# pcd_list = []

# for scaled_pcd in scaled_pcds:
#     # Ensure scaled_pcd is a 2D numpy array with shape (N, 3)
#     scaled_pcd = scaled_pcd.reshape(-1, 3)
    
#     # Create an Open3D point cloud
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(scaled_pcd)
    
#     # Generate a random color
#     color = np.random.rand(3)
#     pcd.colors = o3d.utility.Vector3dVector(np.tile(color, (scaled_pcd.shape[0], 1)))
    
#     # Add the point cloud to the list
#     pcd_list.append(pcd)

# # Visualize all point clouds in the same Open3D window
# o3d.visualization.draw_geometries(pcd_list)


# # pcd1, pcd2, scales = scale_icp_from_depth(depths[0], depths[1], intrinsics[0], intrinsics[1], extrinsics[0], extrinsics[1], max_iterations=100, gt_scales=1/scales)
# pcd1, pcd2, scales = scale_icp(depths[0], depths[1], intrinsics[0], intrinsics[1], extrinsics[0], extrinsics[1], max_iterations=100, gt_scales=1/scales)

    
#     # Create an Open3D point cloud
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(scaled_pcd)
    
#     # Generate a random color
#     color = np.random.rand(3)
#     pcd.colors = o3d.utility.Vector3dVector(np.tile(color, (scaled_pcd.shape[0], 1)))
    
#     # Add the point cloud to the list
#     pcd_list.append(pcd)

# # Visualize all point clouds in the same Open3D window
# o3d.visualization.draw_geometries(pcd_list)


env.shutdown()