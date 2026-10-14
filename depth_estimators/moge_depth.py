import os
import pdb

os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
from depth_estimators.dependency_paths import activate_depth_dependency

activate_depth_dependency('moge')

from typing import *
import itertools
import json
import warnings

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import trimesh
import trimesh.visual
import click

from moge.model.v2 import MoGeModel
from moge.utils.io import save_glb, save_ply
from moge.utils.vis import colorize_depth, colorize_normal
import utils3d
from depth_estimators.base_estimator import BaseDepthEstimator


class MoGeDepth(BaseDepthEstimator):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda")
        self.model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(self.device)
        self.name = 'moge'

    def infer(self, imgs: List[np.ndarray],
              intrinsics=None,
              extrinsics=None) -> Dict[str, Union[np.ndarray, Any]]:
        depths = []
        intrinsics_norm = []
        extrinsics = []

        for img in imgs:
            input_image = img.copy()
            h, w = input_image.shape[:2]
            image_tensor = (
                torch.tensor(input_image / 255.0, dtype=torch.float32, device=self.device)
                .permute(2, 0, 1)
            )

            with torch.no_grad():
                output = self.model.infer(image_tensor)

            depth = output['depth'].cpu().numpy()
            # Filter out invalid depths
            depth[depth < 0] = 0
            depth[depth > 10.0] = 0  # Cap at 10 meters

            intrinsic_norm = output['intrinsics'].cpu().numpy()
            
            depths.append(depth)
            intrinsics_norm.append(intrinsic_norm)

        return {
            "depth": depths,
            # Normalized intrinsics are reported separately from camera poses.
            "intrinsics_norm": intrinsics_norm,
        }


    def get_pcd(self):
        points, depth, mask, intrinsics = self.output['points'].cpu().numpy(), self.output['depth'].cpu().numpy(), self.output[
            'mask'].cpu().numpy(), self.output['intrinsics'].cpu().numpy()
        normals, normals_mask = utils3d.numpy.points_to_normals(points, mask=mask)
        faces, vertices, vertex_colors, vertex_uvs = utils3d.numpy.image_mesh(
            points,
            self.input_image.astype(np.float32) / 255,
            utils3d.numpy.image_uv(width=self.width, height=self.height),
            mask=mask & ~(
                        utils3d.numpy.depth_edge(depth, rtol=0.03, mask=mask) & utils3d.numpy.normals_edge(normals,
                                                                                                        tol=5,
                                                                                                        mask=normals_mask)),
            tri=True
        )
        vertices, vertex_uvs = vertices * [1, -1, -1], vertex_uvs * [1, -1] + [0, 1]

        # Convert to trimesh
        mesh = trimesh.Trimesh(vertices, faces, vertex_colors=vertex_colors, vertex_normals=normals, process=False)

        import open3d as o3d
        # Convert to colored point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vertices)
        pcd.colors = o3d.utility.Vector3dVector(vertex_colors)

        return pcd
