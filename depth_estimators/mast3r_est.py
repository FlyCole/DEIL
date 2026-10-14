from contextlib import contextmanager, redirect_stderr, redirect_stdout
import os
import sys
from pathlib import Path


from depth_estimators.dependency_paths import activate_depth_dependency

activate_depth_dependency('mast3r')


@contextmanager
def _silence_output(enabled):
    if not enabled:
        yield
        return

    with open(os.devnull, "w") as sink:
        with redirect_stdout(sink), redirect_stderr(sink):
            yield


with _silence_output(True):
    from mast3r.model import AsymmetricMASt3R
    from mast3r.fast_nn import fast_reciprocal_NNs
    import mast3r.utils.path_to_dust3r
    from dust3r.utils.device import to_numpy
    from dust3r.utils.image import load_images
    from dust3r.viz import add_scene_cam, CAM_COLORS, OPENGL, pts3d_to_trimesh, cat_meshes
    from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
    from mast3r.cloud_opt.tsdf_optimizer import TSDFPostProcess
    from mast3r.image_pairs import make_pairs

import PIL
import pdb
import copy
import trimesh
import tempfile
import numpy as np
from scipy.spatial.transform import Rotation

from depth_estimators.base_estimator import BaseDepthEstimator

import torchvision.transforms as tvf
ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


class Mast3rEst(BaseDepthEstimator):
    def __init__(self, model_name="naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric",
                 device='cuda', verbose=False, optim_level="coarse",
                 niter1=300, niter2=300):
        """
        Depth estimator using MASt3R model.

        Args:
            model_name (str): Pretrained model name or path to local checkpoint.
            device (str): Device to run the model on.
            verbose (bool): Show MASt3R's internal prints and progress bars.
            optim_level (str): One of ``coarse``, ``refine``, or
                ``refine+depth``. Coarse mode avoids the long fine-alignment
                CUDA kernels that can trip the display GPU watchdog during an
                RLBench rollout.
            niter1 (int): Number of coarse global-alignment iterations.
            niter2 (int): Number of fine iterations when refinement is enabled.
        """
        super().__init__()
        valid_optim_levels = {"coarse", "refine", "refine+depth"}
        if optim_level not in valid_optim_levels:
            raise ValueError(
                f"optim_level must be one of {sorted(valid_optim_levels)}, "
                f"got {optim_level!r}"
            )
        if niter1 < 0 or niter2 < 0:
            raise ValueError("MASt3R iteration counts must be non-negative")

        self.device = device
        self.verbose = verbose
        self.optim_level = optim_level
        self.niter1 = niter1
        self.niter2 = 0 if optim_level == "coarse" else niter2
        with _silence_output(not self.verbose):
            self.model = AsymmetricMASt3R.from_pretrained(model_name).to(device)
        self.name = 'mast3r'

    def infer(self, images, intrinsics=None, extrinsics=None):
        """
        Estimate depth maps for a list of images.

        Args:
            images (list of PIL.Image or np.ndarray): List of input images.
            batch_size (int): Batch size for inference.
            verbose (bool): Whether to print verbose output.
        Returns:
            list of np.ndarray: Estimated depth maps.
        """
        # MASt3R's image loader and sparse global alignment APIs require file
        # paths. Keep those files in a per-call system temporary directory so
        # concurrent inference calls cannot overwrite each other's inputs and
        # everything is removed automatically, including on exceptions.
        with tempfile.TemporaryDirectory(prefix="mast3r_cache_") as output_dir:
            filelist = []
            H, W = images[0].shape[1], images[0].shape[0]
            init = {}

            for i in range(len(images)):
                img = images[i]
                if isinstance(img, np.ndarray):
                    img_pil = PIL.Image.fromarray(img)
                else:
                    img_pil = img
                img_name = f"img_{i}.png"
                img_path = os.path.join(output_dir, img_name)
                filelist.append(img_path)
                img_pil.save(img_path)
                # init.setdefault(img_path, {'intrinsics': intrinsics[i].copy()})

            imgs = load_images(filelist, size=512, verbose=False)
            if len(imgs) == 1:
                imgs = [imgs[0], copy.deepcopy(imgs[0])]
                imgs[1]['idx'] = 1
                filelist = [filelist[0], filelist[0] + '_2']

            scene_graph_params = ["complete"]
            scene_graph = "-".join(scene_graph_params)

            sim_matrix = None
            with _silence_output(not self.verbose):
                pairs = make_pairs(imgs, scene_graph=scene_graph, prefilter=None,
                                   symmetrize=True, sim_mat=sim_matrix)
            # Sparse GA (forward mast3r -> matching -> 3D optimization ->
            # 2D refinement -> triangulation).
            cache_dir = os.path.join(output_dir, "alignment_cache")
            os.makedirs(cache_dir, exist_ok=True)
            optim_level = "refine+depth"
            with _silence_output(not self.verbose):
                scene = sparse_global_alignment(
                    filelist, pairs, cache_dir, self.model, device=self.device,
                    opt_depth="depth" in optim_level, shared_intrinsics=True,
                    init=init, verbose=self.verbose,
                )
            outfile_name = os.path.join(output_dir, "mast3r_recon.glb")

            # TemporaryDirectory owns cleanup, so SparseGAState must not remove
            # any paths independently during object finalization.
            scene_state = SparseGAState(
                scene, False, cache_dir, outfile_name
            )
            pts3d, depth_maps = self.get_3D_model_from_scene(
                False, scene_state, clean_depth=True, as_pointcloud=False
            )

            # Resize depth maps back to the input resolution before the
            # temporary directory and MASt3R's disk cache are deleted.
            depth_maps_resized = []
            for depth_map in depth_maps:
                depth_map_pil = PIL.Image.fromarray(depth_map)
                depth_map_resized_pil = depth_map_pil.resize(
                    (W, H), PIL.Image.BILINEAR
                )
                depth_maps_resized.append(np.array(depth_map_resized_pil))

            return {
                "depth": depth_maps_resized,
                "pts3d": pts3d,
            }
    
    def get_3D_model_from_scene(self, silent, scene_state, min_conf_thr=2, as_pointcloud=False, mask_sky=False,
                            clean_depth=False, transparent_cams=False, cam_size=0.05, TSDF_thresh=0):
        """
        extract 3D_model (glb file) from a reconstructed scene
        """
        if scene_state is None:
            return None
        outfile = scene_state.outfile_name
        if outfile is None:
            return None

        # get optimized values from scene
        scene = scene_state.sparse_ga
        rgbimg = scene.imgs
        focals = scene.get_focals().cpu()
        cams2world = scene.get_im_poses().cpu()

        # 3D pointcloud from depthmap, poses and intrinsics
        if TSDF_thresh > 0:
            tsdf = TSDFPostProcess(scene, TSDF_thresh=TSDF_thresh)
            pts3d, _, confs = to_numpy(tsdf.get_dense_pts3d(clean_depth=clean_depth))
        else:
            pts3d, depthmaps, confs = to_numpy(scene.get_dense_pts3d(clean_depth=clean_depth))
        msk = to_numpy([c > min_conf_thr for c in confs])

        # _convert_scene_output_to_glb(outfile, rgbimg, pts3d, msk, focals, cams2world, as_pointcloud=as_pointcloud,
        #                              transparent_cams=transparent_cams, cam_size=cam_size, silent=silent)

        depth_maps = []
        for i in range(len(depthmaps)):
            depth_map = depthmaps[i]
            depth_map = depth_map.reshape(rgbimg[i].shape[0], rgbimg[i].shape[1])
            depth_pil = PIL.Image.fromarray(depth_map)
            resized_pil = depth_pil.resize((512, 512), PIL.Image.BILINEAR)
            depth_maps.append(np.array(resized_pil))

        return pts3d, depth_maps
        

def _convert_scene_output_to_glb(outfile, imgs, pts3d, mask, focals, cams2world, cam_size=0.05,
                                 cam_color=None, as_pointcloud=False,
                                 transparent_cams=False, silent=False):
    assert len(pts3d) == len(mask) <= len(imgs) <= len(cams2world) == len(focals)
    pts3d = to_numpy(pts3d)
    imgs = to_numpy(imgs)
    focals = to_numpy(focals)
    cams2world = to_numpy(cams2world)

    scene = trimesh.Scene()

    # full pointcloud
    if as_pointcloud:
        pts = np.concatenate([p[m.ravel()] for p, m in zip(pts3d, mask)]).reshape(-1, 3)
        col = np.concatenate([p[m] for p, m in zip(imgs, mask)]).reshape(-1, 3)
        valid_msk = np.isfinite(pts.sum(axis=1))
        pct = trimesh.PointCloud(pts[valid_msk], colors=col[valid_msk])
        scene.add_geometry(pct)
    else:
        meshes = []
        for i in range(len(imgs)):
            pts3d_i = pts3d[i].reshape(imgs[i].shape)
            msk_i = mask[i] & np.isfinite(pts3d_i.sum(axis=-1))
            meshes.append(pts3d_to_trimesh(imgs[i], pts3d_i, msk_i))
        mesh = trimesh.Trimesh(**cat_meshes(meshes))
        mesh.show()
        scene.add_geometry([mesh])

    # add each camera
    for i, pose_c2w in enumerate(cams2world):
        if isinstance(cam_color, list):
            camera_edge_color = cam_color[i]
        else:
            camera_edge_color = cam_color or CAM_COLORS[i % len(CAM_COLORS)]
        add_scene_cam(scene, pose_c2w, camera_edge_color,
                      None if transparent_cams else imgs[i], focals[i],
                      imsize=imgs[i].shape[1::-1], screen_width=cam_size)

    rot = np.eye(4)
    rot[:3, :3] = Rotation.from_euler('y', np.deg2rad(180)).as_matrix()
    scene.apply_transform(np.linalg.inv(cams2world[0] @ OPENGL @ rot))
    if not silent:
        print('(exporting 3D scene to', outfile, ')')
    scene.export(file_obj=outfile)
    return outfile


class SparseGAState:
    def __init__(self, sparse_ga, should_delete=False, cache_dir=None, outfile_name=None):
        self.sparse_ga = sparse_ga
        self.cache_dir = cache_dir
        self.outfile_name = outfile_name
        self.should_delete = should_delete

    def __del__(self):
        if not self.should_delete:
            return
        if self.cache_dir is not None and os.path.isdir(self.cache_dir):
            import shutil
            shutil.rmtree(self.cache_dir)
        self.cache_dir = None
        if self.outfile_name is not None and os.path.isfile(self.outfile_name):
            os.remove(self.outfile_name)
        self.outfile_name = None


def _resize_pil_image(img, long_edge_size):
    S = max(img.size)
    if S > long_edge_size:
        interp = PIL.Image.LANCZOS
    elif S <= long_edge_size:
        interp = PIL.Image.BICUBIC
    new_size = tuple(int(round(x*long_edge_size/S)) for x in img.size)
    return img.resize(new_size, interp)
