import torch
from pytorch3d.ops import sample_farthest_points


def fps_downsample_np(pc, n_sample_point=1024):
    """
    Downsample a point cloud with Furthest Point Sampling (FPS) and return indexes of sampled points.
    :param pc: (N, 3).
    :return:
        fps_idx: (N',).
    """
    pc = torch.from_numpy(pc).unsqueeze(0)
    _, fps_idx = sample_farthest_points(pc, K=n_sample_point)
    fps_idx = fps_idx.numpy()[0]
    return fps_idx


def fps_downsample(pc, n_sample_point=1024):
    """
    Downsample a point cloud with Furthest Point Sampling (FPS) and return indexes of sampled points.
    :param pc: (N, 3) torch.Tensor.
    :return:
        fps_idx: (N',) torch.Tensor.
    """
    pc = pc.unsqueeze(0)
    _, fps_idx = sample_farthest_points(pc, K=n_sample_point)
    fps_idx = fps_idx[0]
    return fps_idx