"""Shared DP3 point-cloud sampling without importing the simulator stack."""

import numpy as np


def farthest_point_sampling(points, num_points=1024, use_cuda=True):
    """Sample XYZ with PyTorch3D and return the corresponding batched indices.

    Torch and PyTorch3D are needed only when sampling. The default deterministic
    starting point and CPU/CUDA behavior match the original DP3 collector.
    """
    import torch
    import pytorch3d.ops as torch3d_ops

    K = [num_points]
    if use_cuda:
        points = torch.from_numpy(points).cuda()
        sampled_points, indices = torch3d_ops.sample_farthest_points(points=points.unsqueeze(0), K=K)
        sampled_points = sampled_points.squeeze(0)
        sampled_points = sampled_points.cpu().numpy()
    else:
        points = torch.from_numpy(points)
        sampled_points, indices = torch3d_ops.sample_farthest_points(points=points.unsqueeze(0), K=K)
        sampled_points = sampled_points.squeeze(0)
        sampled_points = sampled_points.numpy()

    return sampled_points, indices


def preprocess_point_cloud(points, use_cuda=True):
    """Return 1024 XYZ/RGB rows, using XYZ sampling indices for RGB.

    Preserve PyTorch3D's behavior for short clouds: padded XYZ rows are zero and
    their -1 indices select the last input RGB row, as in the original collector.
    """
    points_xyz = points[..., :3]
    points_xyz, sample_indices = farthest_point_sampling(points_xyz, 1024, use_cuda)
    sample_indices = sample_indices.cpu()
    points_rgb = points[sample_indices, 3:][0]
    return np.hstack((points_xyz, points_rgb))
