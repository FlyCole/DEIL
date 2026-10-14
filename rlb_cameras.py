import pdb
from typing import List
from pyrep.objects import VisionSensor, Dummy
import numpy as np
import open3d as o3d
from depth_estimators.base_estimator import BaseDepthEstimator


class RLBCameras():

    def __init__(self,
                 *cameras: VisionSensor,
                 mask: bool = True,
                 depth_estimator=None,
                 mask_rgb_before_dest: bool = False):
        self.cameras = []
        self.mask_cams: List[VisionSensor] = []
        self.segment_out = mask
        if type(cameras[0]) == str:
            self.cameras = [VisionSensor(cam) for cam in cameras]
        for cam in cameras:
            cam.handle_explicitly()
            self.cameras.append(cam)
            name = cam.get_name()
            self.mask_cams.append(VisionSensor(name + '_mask'))
            self.mask_cams[-1].handle_explicitly()

        self.depth_est: BaseDepthEstimator = depth_estimator
        self._estimated_f = [None] * len(self.cameras)
        self.mask_rgb_before_dest = mask_rgb_before_dest
        

        # extr = self.get_extrinsics()
        # for ext in extr:
        #     Dummy.create(0.1).set_matrix(ext)

    
    def change_resolution(self, res_h, res_w):
        for cam in self.cameras:
            cam.set_resolution([res_h, res_w])
            cam.handle_explicitly()
        for cam in self.mask_cams:
            cam.set_resolution([res_h, res_w])
            cam.handle_explicitly()


    def get_masks(self):
        def get_int_mask(cam: VisionSensor):
            # rgb_coded_handles should be (w, h, c)
            # Handle encoded as : handle = R + G * 256 + B * 256 * 256
            
            # If reading from a PIL image, the values will already be on the 
            # range [0, 255] and uint8. However, if rgb_coded_handles is read directly from 
            # a sensor, it will be on the range [0, 1].
            rgb_coded_handles = cam.capture_rgb()
            if rgb_coded_handles.dtype != np.uint8:
                rgb_coded_handles *= 255  # takes rgb range to 0 -> 255
            rgb_coded_handles.astype(int)
            return (rgb_coded_handles[:, :, 0] +
                    rgb_coded_handles[:, :, 1] * 256 +
                    rgb_coded_handles[:, :, 2] * 256 * 256)
        masks = [get_int_mask(cam) for cam in self.mask_cams]
        return [(mask > 60) * (mask < 4000) for mask in masks]

    def get_rgbs(self):
        return [cam.capture_rgb() for cam in self.cameras]
    
    def get_gt_depths(self):
        return [cam.capture_depth(in_meters=True) for cam in self.cameras]

    def get_depths(self, force_gt=False, masks: List[np.ndarray] = None):
        if self.depth_est is None or force_gt:
            self._estimated_f = [None] * len(self.cameras)
            return self.get_gt_depths()
        depth_data = self.get_estimated_depth_data(masks=masks)
        if "focallength_px" in depth_data[0].keys():
            self._estimated_f = [data['focallength_px'] for data in depth_data] #TODO: check if this makes sense to use
        return [data['depth'] for data in depth_data]
    
    def get_estimated_depth_data(self, masks: List[np.ndarray] = None):
        rgbs = self.get_rgbs()
        if masks is not None:
            rgbs = [rgb * mask[:, :, None] for rgb, mask in zip(rgbs, masks)]
        return [self.depth_est.infer(rgb) for rgb in rgbs]
    
    def get_extrinsics(self):
        return [cam.get_matrix() for cam in self.cameras]
    
    def get_intrinsics(self):
        gt_Ks = [cam.get_intrinsic_matrix() for cam in self.cameras]
        if self._estimated_f[0] is not None:
            for i in range(len(gt_Ks)):
                gt_Ks[i][0, 0] = - self._estimated_f[i]
                gt_Ks[i][1, 1] = - self._estimated_f[i]
        return gt_Ks
    
    def get_gt_pcds(self):
        return [cam.capture_pointcloud().reshape(-1, 3) for cam in self.cameras]
    
    def get_gt_scene_pcds(self):
        pcds = self.get_gt_pcds()
        return np.concatenate(pcds, axis=1)
    
    def get_gt_full_scene_pcd(self):
        segment_out = self.segment_out
        self.segment_out = False
        pcds = self.get_gt_pcds()
        self.segment_out = segment_out
        return np.concatenate(pcds, axis=0)
    
    def get_estimated_full_scene_pcd(self):
        segment_out = self.segment_out
        self.segment_out = False
        pcds = self.get_estimated_pcd()
        self.segment_out = segment_out
        return np.concatenate(pcds, axis=0)
    
    @staticmethod
    def reproject_to_pcd(depth, intrinsics, extrinsics, mask=None):
        # depth is camera-axis z in meters; extrinsics maps camera to world.
        # Keep RLBench's signed focal lengths when unprojecting its images.
            
        h, w = depth.shape
        i, j = np.meshgrid(np.arange(w), np.arange(h), indexing='xy')
        # Match demonstration preprocessing: never mutate the caller's depth
        # array, and prevent invalid or behind-camera depths from reaching the
        # point-cloud reconstruction.
        z = np.asarray(depth, dtype=np.float32).copy()
        z[~np.isfinite(z)] = 0.0
        z[z < 0.0] = 0.0
        x = (i - intrinsics[0, 2]) * z / intrinsics[0, 0]
        y = (j - intrinsics[1, 2]) * z / intrinsics[1, 1]

        pcd_camera = np.stack((x, y, z), axis=-1)
        if mask is not None:
            pcd_camera = pcd_camera[mask]
        pcd_camera = pcd_camera.reshape(-1, 3)
        
        # Convert to homogeneous coordinates
        pcd_camera_hom = np.concatenate([pcd_camera, np.ones((pcd_camera.shape[0], 1))], axis=1)
        
        # Reproject to world frame
        pcd_world_hom = (extrinsics @ pcd_camera_hom.T).T
        
        # Convert back to non-homogeneous coordinates
        pcd_world = pcd_world_hom[:, :3]
        
        return pcd_world
        # return VisionSensor.pointcloud_from_depth_and_camera_params(depth, extrinsics, intrinsics)

    @staticmethod
    def reproject_to_pcd_o3d(depth, intrinsics, extrinsics, mask=None):
        if mask is not None:
            depth = depth * mask

        pcd_o3d = o3d.geometry.PointCloud.create_from_depth_image(
            o3d.geometry.Image(depth),
            o3d.camera.PinholeCameraIntrinsic(
                width=depth.shape[1],
                height=depth.shape[0],
                fx=intrinsics[0, 0],
                fy=intrinsics[1, 1],
                cx=intrinsics[0, 2],
                cy=intrinsics[1, 2],
            ),
            extrinsic=np.linalg.inv(extrinsics),
            depth_scale=1,
        )
        return pcd_o3d
      
    def get_reprojected_pcd(self, depths: List[np.ndarray], masks: List[np.ndarray] = None):
        intrinsics = self.get_intrinsics()
        extrinsics = self.get_extrinsics()
        if masks is None:
            masks = [None] * len(self.cameras)
        if self.segment_out and masks[0] is None:
            masks = self.get_masks()
        reprojected_pcds = [self.reproject_to_pcd(depths[i], intrinsics[i], extrinsics[i], masks[i]) for i in range(len(depths))]
        return reprojected_pcds
    
    def get_single_gt_reprojected_pcd(self):
        depths = self.get_depths(force_gt=True)
        reprojected_pcds = self.get_reprojected_pcd(depths)
        return np.concatenate(reprojected_pcds, axis=0)
    
    def get_gt_reprojected_pcd(self):
        depths = self.get_depths(force_gt=True)
        reprojected_pcds = self.get_reprojected_pcd(depths)
        return reprojected_pcds
    
    def get_estimated_pcd(self):
        masks=None
        if self.mask_rgb_before_dest:
            masks = self.get_masks()
        depths = self.get_depths(masks=masks)
        reprojected_pcds = self.get_reprojected_pcd(depths, masks=masks)
        return reprojected_pcds
    
    def get_scaled_pcd(self, force_gt: bool = False):
        depths = self.get_depths(force_gt=force_gt)
        #scale the depth by random numbers one for each camera
        scales = np.random.uniform(0.2, 9, len(self.cameras))
        print(f"GT scales: {scales}")
        depths = [depth * scale for depth, scale in zip(depths, scales)]
        reprojected_pcds = self.get_reprojected_pcd(depths)
        return reprojected_pcds, scales

    def get_pcds_from_moge(self):
        estimated_pcds = []
        for rgb in self.get_rgbs():
            self.depth_est.infer(rgb)
            pcd = self.depth_est.get_pcd()

            estimated_pcds.append(pcd)

        return estimated_pcds
