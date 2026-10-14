from typing import Dict, Union, Any, List, Tuple
import numpy as np

import torch
from PIL import Image
from torchvision import transforms as TF
from depth_estimators.base_estimator import BaseDepthEstimator
from depth_estimators.dependency_paths import activate_depth_dependency

activate_depth_dependency('vggt')
from vggt.models.vggt import VGGT as VGGT_Model
from utils import Tmtx_from_posevec

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


def preprocess_images(images_pre: List[np.ndarray], mode="crop"):
    """
    A quick start function to load and preprocess images for model input.
    This assumes the images should have the same shape for easier batching, but our model can also work well with different shapes.

    Args:
        image_path_list (list): List of paths to image files
        mode (str, optional): Preprocessing mode, either "crop" or "pad".
                             - "crop" (default): Sets width to 518px and center crops height if needed.
                             - "pad": Preserves all pixels by making the largest dimension 518px
                               and padding the smaller dimension to reach a square shape.

    Returns:
        torch.Tensor: Batched tensor of preprocessed images with shape (N, 3, H, W)

    Raises:
        ValueError: If the input list is empty or if mode is invalid

    Notes:
        - Images with different dimensions will be padded with white (value=1.0)
        - A warning is printed when images have different shapes
        - When mode="crop": The function ensures width=518px while maintaining aspect ratio
          and height is center-cropped if larger than 518px
        - When mode="pad": The function ensures the largest dimension is 518px while maintaining aspect ratio
          and the smaller dimension is padded to reach a square shape (518x518)
        - Dimensions are adjusted to be divisible by 14 for compatibility with model requirements
    """

    if type(images_pre) == np.ndarray:
        images_pre = [images_pre]
    
    # Validate mode
    if mode not in ["crop", "pad"]:
        raise ValueError("Mode must be either 'crop' or 'pad'")
    

    images = []
    pad_masks = []
    image_info = []
    shapes = set()
    to_tensor = TF.ToTensor()
    target_size = 518

    # First process all images and collect their shapes
    for img_np in images_pre:
        # Open image
        # img = Image.fromarray((img_np * 255).astype(np.uint8))
        img = Image.fromarray(img_np.astype(np.uint8))

        # If there's an alpha channel, blend onto white background:
        if img.mode == "RGBA":
            # Create white background
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            # Alpha composite onto the white background
            img = Image.alpha_composite(background, img)

        # Now convert to "RGB" (this step assigns white for transparent areas)
        img = img.convert("RGB")

        width, height = img.size
        i_info = {"og_size_wh": (width, height),
                  "target_size": target_size,
                  "mode": mode}
        
        if mode == "pad":
            # Make the largest dimension 518px while maintaining aspect ratio
            if width >= height:
                new_width = target_size
                new_height = round(height * (new_width / width) / 14) * 14  # Make divisible by 14
            else:
                new_height = target_size
                new_width = round(width * (new_height / height) / 14) * 14  # Make divisible by 14
        else:  # mode == "crop"
            # Original behavior: set width to 518px
            new_width = target_size
            # Calculate height maintaining aspect ratio, divisible by 14
            new_height = round(height * (new_width / width) / 14) * 14

        # Resize with new dimensions (width, height)
        i_info["new_size_wh"] = (new_width, new_height)
        img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        img = to_tensor(img)  # Convert to tensor (0, 1)

        # Center crop height if it's larger than 518 (only in crop mode)
        if mode == "crop" and new_height > target_size:
            start_y = (new_height - target_size) // 2
            img = img[:, start_y : start_y + target_size, :]
            i_info["height_crop_start"] = start_y
        
        # For pad mode, pad to make a square of target_size x target_size
        pad_mask = torch.ones((img.shape[0], img.shape[1]), dtype=torch.bool)
        if mode == "pad":
            h_padding = target_size - img.shape[1]
            w_padding = target_size - img.shape[2]
            
            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left
                
                # Pad with white (value=1.0)
                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )
                pad_mask = torch.nn.functional.pad(
                    pad_mask, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0
                )
                print(pad_mask)

        shapes.add((img.shape[1], img.shape[2]))
        images.append(img)
        pad_masks.append(pad_mask)
        image_info.append(i_info)

    # Check if we have different shapes
    # In theory our model can also work well with different shapes
    if len(shapes) > 1:
        print(f"Warning: Found images with different shapes: {shapes}")
        # Find maximum dimensions
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)

        # Pad images if necessary
        padded_images = []
        second_pad_masks = []
        for img, pad_mask in zip(images, pad_masks):
            h_padding = max_height - img.shape[1]
            w_padding = max_width - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )
                pad_mask = torch.nn.functional.pad(
                    pad_mask, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0
                )
            padded_images.append(img) #TODO: Certainly better way to do this (even just enumerate and overwrite)
            second_pad_masks.append(pad_mask)
        images = padded_images
        pad_masks = second_pad_masks

    images = torch.stack(images)  # concatenate images
    pad_masks = torch.stack(pad_masks)  # concatenate pad

    # Ensure correct shape when single image
    if len(images_pre) == 1:
        # Verify shape is (1, C, H, W)
        if images.dim() == 3:
            images = images.unsqueeze(0)
            pad_masks = pad_masks.unsqueeze(0)

    return images, pad_masks, image_info


def postprocess_depth(depth_imgs: torch.Tensor, image_info: List[dict]):
    """
    Postprocess the depth map to match the original image size.

    Args:
        depth (torch.Tensor): Depth map tensor with shape (N, 1, H, W)
        image_info (dict): Image information dictionary

    Returns:
        torch.Tensor: Postprocessed depth map tensor with shape (N, 1, H, W)
    """
    processed_depths = []

    for iidx, i_info in enumerate(image_info):
        depth = depth_imgs[:, iidx, :, :, 0]
        if i_info["mode"] == "crop":
            og_width, og_height = i_info["og_size_wh"]
            final_width, final_height = depth.shape[0], depth.shape[1]
            resized__width, resized_height = i_info["new_size_wh"]
            target_size = i_info["target_size"]
            start_y = i_info.get("height_crop_start", 0)

            depth = torch.nn.functional.pad(
                    depth, (0, 0, 0, 0, start_y, resized_height - start_y - final_height), mode="constant", value=0.0
                )
            
            # assert depth.shape[1] == resized_height, f"Expected height {resized_height}, got {depth.shape[1]}"
            depth.unsqueeze_(0)
            depth = torch.nn.functional.interpolate(depth, size=(og_height, og_width), mode="nearest")

        else:
            pass
        processed_depths.append(depth[0,0].cpu().numpy())

    return np.stack(processed_depths)


def Kmtx_from_fov(fov: np.ndarray,
                  image_size: Tuple[int, int]):
    """
    Convert a field of view (FOV) to a camera intrinsic matrix.
    Assuming camera's principal point is at the image center
    Args:
        fov (np.ndarray): Field of view in radians, shape (2,) for horizontal and vertical FOV.
        image_size (Tuple[int, int]): Size of the image as (height, width).
    Returns:
        np.ndarray: Camera intrinsic matrix K of shape (3, 3).
    """
    K = np.eye(3, dtype=np.float32)
    height, width = image_size
    K[0, 0] = width / (2 * np.tan(fov[0] / 2))  # fx
    K[1, 1] = height / (2 * np.tan(fov[1] / 2))  # fy
    K[0, 2] = width / 2  # cx
    K[1, 2] = height / 2  # cy
    return K


class VGGT(BaseDepthEstimator):

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
        self.dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        self.model = VGGT_Model.from_pretrained("facebook/VGGT-1B").to(self.device)
        self.name = "vggt"
        
    def single_infer(self,
              img: np.ndarray,
              mask: np.ndarray = None,
              **kwargs) -> Dict[str, Union[np.ndarray, Any]]:
        """Run inference on an image.

        Args:
        ----
            img: The image loaded as a numpy array.

        Returns:
        -------
            prediction: A dictionary containing the following keys:
                depth: estimated depth image
        """
        images, pad_masks, img_info = preprocess_images(img, mode="crop")

        images = images.to(self.device)
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=self.dtype):
                # Predict attributes including cameras, depth maps, and point maps.
                predictions = self.model(images)

        depths = postprocess_depth(predictions["depth"], img_info)


        # Get the predicted camera extrinsics
        T_WCs = predictions["pose_enc"][0,:,:7].cpu().numpy()
        T_WCs = [
            Tmtx_from_posevec(t = posevec[4:], q = posevec[:4])
            for posevec in T_WCs
        ]

        # Get the predicted camera intrinsics
        Ks = predictions["pose_enc"][0,:,7:].cpu().numpy()
        Ks = [
            Kmtx_from_fov(fov = K[:2], image_size = img[0].shape[:2])
            for K in Ks
        ]

        if type(img) == np.ndarray:
            # If only one image was processed, ensure depth is 2D
            depths = depths[0]
        return {
            "depth": depths, #TODO: This defeats the purpose of the batch
            "confidence": predictions["depth_conf"][0, 0, :, :].cpu().numpy(),
            "extrinsics": T_WCs,
            "intrinsics": Ks,
        }
    
    def infer(self,
              imgs: List[np.ndarray],
              intrinsics: List[np.ndarray] = None,
              extrinsics: List[np.ndarray] = None,
              masks: List[np.ndarray] = None,
              **kwargs) -> Dict[str, Union[np.ndarray, Any]]:
        """Run inference on multiple images.
        Args:
        ----
            imgs: A list of images loaded as numpy arrays.
        Returns:
        -------
            prediction: A dictionary containing the following keys:
                depth: estimated depth images
        """
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        from vggt.utils.geometry import unproject_depth_map_to_point_map
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=self.dtype):
                images, _, img_info = preprocess_images(imgs, mode="crop")
                images = images[None].to(self.device)  # Add batch dimension
                aggregated_tokens_list, ps_idx = self.model.aggregator(images)

            # Predict Cameras
            pose_enc = self.model.camera_head(aggregated_tokens_list)[-1]
            # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
            extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])

            # Predict Depth Maps
            depth_map, depth_conf = self.model.depth_head(aggregated_tokens_list, images, ps_idx)
            # Filter based on confidence
            # depth_map[depth_conf < 1.5] = 0.0
            # depth_map[depth_conf < 0.8] = 0.0
            depths = postprocess_depth(depth_map, img_info)

            # Predict Point Maps
            point_map, point_conf = self.model.point_head(aggregated_tokens_list, images, ps_idx)
                
            # Construct 3D Points from Depth Maps and Cameras
            # which usually leads to more accurate 3D points than point map branch
            point_map_by_unprojection = unproject_depth_map_to_point_map(depth_map.squeeze(0), 
                                                                        extrinsic.squeeze(0), 
                                                                        intrinsic.squeeze(0))

            # Predict Tracks
            # choose your own points to track, with shape (N, 2) for one scene
            query_points = torch.FloatTensor([[100.0, 200.0], 
                                                [60.72, 259.94]]).to(self.device)
            track_list, vis_score, conf_score = self.model.track_head(aggregated_tokens_list, images, ps_idx, query_points=query_points[None])

        return {
            "depth": depths,
            # "depth": depth_map.squeeze(0).cpu().numpy(),
            "confidence": depth_conf.squeeze(0).cpu().numpy(),
            "extrinsics": extrinsic.squeeze(0).cpu().numpy(),
            "intrinsics": intrinsic.squeeze(0).cpu().numpy(),
            "point_map": point_map.squeeze(0).cpu().numpy(),
            "point_map_by_unprojection": point_map_by_unprojection,
        }
    
    def __delete__(self):
        """
        Clean up the model and free resources.
        """
        del self.model
        torch.cuda.empty_cache()
        print("VGGT model deleted and resources freed.")
