from depth_estimators.base_estimator import BaseDepthEstimator
import torch
import os
import sys
import PIL
import pdb
import tempfile
import numpy as np
from depth_estimators.dependency_paths import activate_croco, activate_depth_dependency

activate_croco()
activate_depth_dependency('pi3')

from pi3.models.pi3 import Pi3
from pi3.utils.basic import load_images_as_tensor, write_ply
from pi3.utils.geometry import depth_edge


class Pi3Est(BaseDepthEstimator):
    def __init__(self):
        self.name = "pi3"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = Pi3.from_pretrained("yyfz233/Pi3").to(self.device).eval()

    def infer(self, images, intrinsics=None, extrinsics=None):
        # 1. Save images in a unique temporary folder so concurrent Pi3
        # inference processes cannot overwrite each other's cache.
        with tempfile.TemporaryDirectory(prefix="pi3_cache_") as cache_dir:
            for i in range(len(images)):
                img = images[i]
                if isinstance(img, np.ndarray):
                    img_pil = PIL.Image.fromarray(img)
                else:
                    img_pil = img
                img_name = f"img_{i}.png"
                img_path = os.path.join(cache_dir, img_name)
                img_pil.save(img_path)

            # 2. Prepare input data
            # The load_images_as_tensor function will print the loading path
            imgs = load_images_as_tensor(cache_dir, interval=1).to(self.device) # (N, 3, H, W)

        # Inference currently requires CUDA, including its autocast context.
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=dtype):
                res = self.model(imgs[None]) # Add batch dimension
        
        # Confidence and edge masks filter the returned global point cloud.
        masks = torch.sigmoid(res['conf'][..., 0]) > 0.1
        non_edge = ~depth_edge(res['local_points'][..., 2], rtol=0.03)
        masks = torch.logical_and(masks, non_edge)[0]

        point_maps = res['local_points'][0]
        depths = []
        for point_map in point_maps:
            depth_map = point_map[..., 2].cpu().numpy()

            # Resize back to original image size
            H_orig, W_orig = images[0].shape[0], images[0].shape[1]
            depth_map = PIL.Image.fromarray(depth_map).resize((W_orig, H_orig), PIL.Image.BILINEAR)
            depth_map = np.array(depth_map)
            depths.append(depth_map)
            
        results = {
            'depth': depths,  # (H, W)
            'pts3d': res['points'][0][masks].cpu().numpy(), # (N, 3)
        }


        return results
