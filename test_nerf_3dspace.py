import os
import os.path as osp
import yaml
import argparse
import numpy as np
import imageio
import open3d as o3d

import torch
import torch.nn as nn

from model import FourierEmbedding, D_NeRF
from render import point_query
from utils.repro_util import load_official_dnerf_weights
from utils.visual_util import build_colored_pointcloud


# Create tensor on GPU by default ('.to(device)' & '.cuda()' cost time!)
torch.set_default_tensor_type('torch.cuda.FloatTensor')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Config files')
    parser.add_argument('--checkpoint', type=int, default=200000, help='Checkpoint (iteration) to load')

    # Read parameters
    args = parser.parse_args()
    with open(args.config) as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
    for ckey, cvalue in configs.items():
        args.__dict__[ckey] = cvalue

    # Fix the random seed
    seed = args.random_seed
    np.random.seed(seed)
    torch.manual_seed(seed)


    # # Load the data
    # if args.dataset_type == 'blender':
    #     from data_loader.load_blender import load_blender_data
    #     imgs_test, poses_test, times_test, img_h, img_w, focal = load_blender_data(data_root=args.data_root,
    #                                                                                split='test',
    #                                                                                half_res=args.half_res,
    #                                                                                white_bkgd=args.white_bkgd)
    #     n_view_test = imgs_test.shape[0]
    #
    #     near = 2.
    #     far = 6.
    #
    # else:
    #     raise ValueError('Not implemented!')

    # Create the Fourier embedding
    point_embedding = FourierEmbedding(n_freq=args.n_freq_point)
    view_embedding = FourierEmbedding(n_freq=args.n_freq_view)
    time_embedding = FourierEmbedding(n_freq=args.n_freq_time, input_dim=1)

    # Create the network (coarse) and load trained model weights
    model = D_NeRF(n_layer=args.n_layer,
                   n_dim=args.n_dim,
                   input_dim=point_embedding.output_dim,
                   input_view_dim=view_embedding.output_dim,
                   input_time_dim=time_embedding.output_dim,
                   skips=[4],
                   use_viewdir=args.use_viewdir,
                   embedding=point_embedding,
                   rgb_act=args.rgb_act,
                   density_act=args.density_act)
    load_pretrained = (args.checkpoint < 0)
    if load_pretrained:
        weight_path = 'pretrained/%s/800000.tar'%(args.dataset_name)
        model.load_state_dict(load_official_dnerf_weights(weight_path))
    else:
        weight_path = osp.join(args.exp_base, 'model_%06d.pth.tar'%(args.checkpoint))
        model.load_state_dict(torch.load(weight_path))

    # Create the network (fine) and load trained model weights
    fine_sampling = (args.n_sample_point_fine > 0)
    model_fine = None
    if fine_sampling and args.two_model_for_fine:
        model_fine = D_NeRF(n_layer=args.n_layer,
                            n_dim=args.n_dim,
                            input_dim=point_embedding.output_dim,
                            input_view_dim=view_embedding.output_dim,
                            input_time_dim=time_embedding.output_dim,
                            skips=[4],
                            use_viewdir=args.use_viewdir,
                            embedding=point_embedding,
                            rgb_act=args.rgb_act,
                            density_act=args.density_act)
        weight_path_fine = osp.join(args.exp_base, 'model_%06d_fine.pth.tar'%(args.checkpoint))
        model_fine.load_state_dict(torch.load(weight_path_fine))

    # Create path to save rendered images
    exp_base = args.exp_base
    save_render_base = osp.join(exp_base, 'test_%06d_3dspace'%(args.checkpoint))
    os.makedirs(save_render_base, exist_ok=True)


    """
    Directly observe the NeRF density at 3D space points
    """
    volume_size = 4.0
    volume_sample = 200
    points = torch.linspace(-0.5*volume_size, 0.5*volume_size, steps=volume_sample+1)
    points = torch.stack(torch.meshgrid([points, points, points]), -1)
    points = points.reshape(-1, 3)
    cur_time = torch.Tensor([0.0])

    # Batchify
    rgb, density = [], []
    for i in range(0, points.shape[0], args.chunk_point):
        # Forward
        with torch.no_grad():
            points_batch = points[i:(i + args.chunk_point)]
            viewdirs_batch = torch.ones_like(points_batch)       # viewdirs is not used, feed any value is OK
            viewdirs_batch = viewdirs_batch / viewdirs_batch.norm(dim=1, keepdim=True)
            rgb_batch, density_batch, _ = point_query(points_batch, viewdirs_batch, cur_time,
                                                      point_embedding, view_embedding, time_embedding,
                                                      model, model_fine,
                                                      fine_sampling=fine_sampling,
                                                      zero_canonical=True)
            rgb.append(rgb_batch)
            density.append(density_batch)

    rgb = torch.cat(rgb, 0)
    rgb = rgb.reshape([volume_sample+1, volume_sample+1, volume_sample+1, 3]).cpu().numpy()
    density = torch.cat(density, 0)
    alpha = 1. - torch.exp(- density * volume_size / volume_sample)
    alpha = alpha.reshape([volume_sample+1, volume_sample+1, volume_sample+1]).cpu().numpy()
    points = points.reshape([volume_sample+1, volume_sample+1, volume_sample+1, 3]).cpu().numpy()

    """
    Visualize non-empty 3D points
    """
    # Filter out empty points
    alpha_thresh = 0.5
    valid = (alpha > alpha_thresh)
    points_sel, alpha_sel, rgb_sel = points[valid], alpha[valid], rgb[valid]

    # # Check density range
    # import matplotlib.pyplot as plt
    # plt.plot(np.sort(alpha))
    # plt.show()

    # Visualize
    pcds = []
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    pcds.append(coord_frame)
    color = 1 - np.expand_dims(alpha_sel, 1) * np.ones_like(points_sel)
    color = color.clip(0., 1.)
    pcds.append(build_colored_pointcloud(points_sel, color))
    rgb_sel = rgb_sel.clip(0., 1.)
    pcds.append(build_colored_pointcloud(points_sel, rgb_sel).translate([volume_size, 0., 0.]))
    o3d.visualization.draw_geometries(pcds)

    """
    # Visualize 2D slice planes
    # """
    # for t in range(points.shape[2]):
    #     alpha_map = alpha[:, :, t]
    #     save_path = osp.join(save_render_base, '%06d.png' % (t))
    #     imageio.imwrite(save_path, alpha_map)