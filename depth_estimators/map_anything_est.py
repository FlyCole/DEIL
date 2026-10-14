# Optional config for better memory efficiency
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from typing import Dict, List, Optional
import numpy as np
from depth_estimators.base_estimator import BaseDepthEstimator


RLBench_TO_OPENCV_CAMERA = np.array(
    [[-1, 0, 0, 0],
     [0, -1, 0, 0],
     [0, 0, 1, 0],
     [0, 0, 0, 1]],
    dtype=np.float32,
)


class MapAnythingEst(BaseDepthEstimator):
    """MapAnything with known RLBench poses, or the paper's no-pose ablation."""

    def __init__(self, use_camera_poses: bool = True):
        if not isinstance(use_camera_poses, bool):
            raise ValueError("use_camera_poses must be a boolean")

        # Defer optional upstream dependencies until this estimator is selected.
        from depth_estimators.dependency_paths import activate_depth_dependency

        activate_depth_dependency('map_anything')
        from mapanything.models import MapAnything
        from mapanything.utils.image import preprocess_inputs

        self.use_camera_poses = use_camera_poses
        self.name = "map_anything" if use_camera_poses else "map_anything_no_ext"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._preprocess_inputs = preprocess_inputs

        # Weights are fetched from Hugging Face on first use; later runs use
        # the local Hub cache. The selected checkpoint has its own license.
        self.model = MapAnything.from_pretrained("facebook/map-anything").to(self.device)


    def aggregate_inputs(self, 
                         rgbs: List[np.ndarray], 
                         intrinsics: List[np.ndarray], 
                         extrinsics: Optional[List[np.ndarray]] = None):
        """Build upstream inputs without changing the supplied calibration.

        The standard variant receives metric camera-to-world poses. The no-pose
        variant ignores ``extrinsics``, even when the caller supplies them.
        """
        if rgbs is None or len(rgbs) == 0:
            raise ValueError("At least one RGB view is required")
        if intrinsics is None or len(intrinsics) != len(rgbs):
            raise ValueError("Each RGB view must have a corresponding intrinsic matrix")
        if self.use_camera_poses and (
            extrinsics is None or len(extrinsics) != len(rgbs)
        ):
            raise ValueError("Known-pose MapAnything requires one extrinsic matrix per RGB view")

        views = []
        
        for i in range(len(rgbs)):
            rgb = np.asarray(rgbs[i])
            if rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) == 0:
                raise ValueError(f"RGB view {i} must have shape (H, W, 3)")

            intrinsics_cv = np.asarray(intrinsics[i], dtype=np.float32).copy()
            if intrinsics_cv.shape != (3, 3) or not np.isfinite(intrinsics_cv).all():
                raise ValueError(f"Intrinsic matrix {i} must be a finite (3, 3) matrix")
            if intrinsics_cv[0, 0] == 0 or intrinsics_cv[1, 1] == 0:
                raise ValueError(f"Intrinsic matrix {i} must have nonzero focal lengths")
            intrinsics_cv[0, 0] = abs(intrinsics_cv[0, 0])
            intrinsics_cv[1, 1] = abs(intrinsics_cv[1, 1])

            view = {
                "img": rgb,                     # (H, W, 3) - [0, 255]
                "intrinsics": intrinsics_cv,     # (3, 3), OpenCV positive focal lengths
            }
            if self.use_camera_poses:
                camera_pose = np.asarray(extrinsics[i], dtype=np.float32)
                if camera_pose.shape != (4, 4) or not np.isfinite(camera_pose).all():
                    raise ValueError(f"Extrinsic matrix {i} must be a finite (4, 4) matrix")
                if not np.allclose(camera_pose[3], [0, 0, 0, 1]):
                    raise ValueError(f"Extrinsic matrix {i} must be a homogeneous camera-to-world pose")

                # RLBench camera axes are left/up/forward. Right-multiplication
                # converts them to OpenCV right/down/forward in the same world
                # frame and leaves the metric camera translation unchanged.
                view["camera_poses"] = camera_pose @ RLBench_TO_OPENCV_CAMERA
                view["is_metric_scale"] = True
            
            views.append(view)
        return views


    def infer(self, rgbs, intrinsics=None, extrinsics=None) -> Dict[str, np.ndarray]:
        """Estimate geometry using the selected paper variant.

        Args:
            rgbs: List of uint8 RGB arrays with shape (H, W, 3).
            intrinsics: Corresponding (3, 3) RLBench calibration matrices.
            extrinsics: Corresponding metric (4, 4) camera-to-world poses.
                Required for ``map_anything``; ignored for ``map_anything_no_ext``.

        Returns:
            A dictionary with per-view lists: ``depth`` contains camera-axis
            z-depth resized to the input resolution; ``pts3d`` contains flattened
            world point maps; ``camera_poses`` contains predicted OpenCV poses.
            The depth outputs are capped at median + 3 * MAD before returning.
        """
        views = self.aggregate_inputs(rgbs, intrinsics, extrinsics)

        # Preprocess inputs to the expected format
        processed_views = self._preprocess_inputs(views)

        # Run inference with any combination of inputs
        predictions = self.model.infer(
            processed_views,                  # Any combination of input views
            memory_efficient_inference=False, # Trades off speed for more views (up to 2000 views on 140 GB)
            use_amp=self.device == "cuda",    # Mixed precision is only supported on CUDA upstream
            amp_dtype="bf16",                 # bf16 inference (recommended; falls back to fp16 if bf16 not supported)
            apply_mask=True,                  # Apply masking to dense geometry outputs
            mask_edges=True,                  # Remove edge artifacts by using normals and depth
            apply_confidence_mask=False,      # Filter low-confidence regions
            confidence_percentile=10,         # Remove bottom 10 percentile confidence pixels
            ignore_calibration_inputs=False,
            ignore_depth_inputs=False,
            ignore_pose_inputs=not self.use_camera_poses,
            ignore_depth_scale_inputs=False,
            ignore_pose_scale_inputs=not self.use_camera_poses,
        )

        # Access results for each view - Complete list of metric outputs
        results = {
            'depth': [],
            'pts3d': [],
            'camera_poses': [],
        }
        for i, pred in enumerate(predictions):
            # Geometry outputs
            pts3d = pred["pts3d"]                     # 3D points in world coordinates (B, H, W, 3)
            results['pts3d'].append(pts3d[0].reshape(-1, 3).cpu().numpy())
            depth_z = pred["depth_z"]                 # Z-depth in camera frame (B, H, W, 1)

            # Resize the depth back to the original image size and convert to numpy
            depth_resized = torch.nn.functional.interpolate(
                depth_z.permute(0, 3, 1, 2),
                size=views[i]['img'].shape[:2],
                mode='bilinear',
                align_corners=False
            ).permute(0, 2, 3, 1)
            depth_z = depth_resized

            # Remove depth outliers: clamp to median + 3*MAD
            dz = depth_z[0, :, :, 0]
            median = dz.median()
            mad = (dz - median).abs().median()
            upper = median + 3 * mad
            dz = dz.clamp(max=upper.item())
            depth_z[0, :, :, 0] = dz

            results['depth'].append(depth_z[0, :, :, 0].cpu().numpy())
            camera_poses = pred["camera_poses"]       # OpenCV (+X - Right, +Y - Down, +Z - Forward) cam2world poses in world frame (B, 4, 4)
            results['camera_poses'].append(camera_poses[0].cpu().numpy())

        return results
