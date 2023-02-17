import os
import os.path as osp
from tqdm import tqdm
import yaml
import argparse
import numpy as np
import open3d as o3d

import torch
import torch.nn as nn

from model import FourierEmbedding, D_NeRF
from camera import Camera, Rays, convert_rays_to_ndc, restore_ndc_points
from render import nerf_render, point_query
from utils.repro_util import load_official_dnerf_weights
from utils.point_util import fps_downsample
from utils.visual_util import build_colored_pointcloud, pc_flow_to_sphere, visualize_point_flow_plt


# Create tensor on GPU by default ('.to(device)' & '.cuda()' cost time!)
torch.set_default_tensor_type('torch.cuda.FloatTensor')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Config files')
    parser.add_argument('--checkpoint', type=int, default=800000, help='Checkpoint (iteration) to load')

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


    # Load the data
    if args.dataset_type == 'blender':
        from data_loader.load_blender import load_blender_data
        imgs_test, poses_test, times_test, img_h, img_w, focal = load_blender_data(data_root=args.data_root,
                                                                                   split='test',
                                                                                   half_res=args.half_res,
                                                                                   white_bkgd=args.white_bkgd)
        n_view_test = imgs_test.shape[0]

        near = 2.
        far = 6.

    else:
        raise ValueError('Not implemented!')

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

    # Create path to save visualization results
    exp_base = args.exp_base
    if load_pretrained:
        save_render_base = osp.join(exp_base, 'test_pretrained_trajectory')
    else:
        save_render_base = osp.join(exp_base, 'test_%06d_trajectory'%(args.checkpoint))
    os.makedirs(save_render_base, exist_ok=True)


    n_sample_point = 1024   # 64
    track_box_size = 0.05
    track_box_sample = 10
    track_times = torch.linspace(0., 1., steps=10)
    vid = 6     # Select a proper view to study the sequence
    pose = torch.Tensor(poses_test[vid])

    """
    Get initial points from 1st frame
    """
    # Give a time step to query
    cur_time = track_times[:1]

    # Get rays for all pixels
    cam = Camera(img_h, img_w, focal, pose)
    rays_o, rays_d = cam.get_rays()
    rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)
    viewdirs = rays_d / rays_d.norm(dim=1, keepdim=True)
    if args.use_ndc:
        rays_o, rays_d = convert_rays_to_ndc(rays_o, rays_d, img_h, img_w, focal, near_plane=1.)

    # Batchify
    points, points_fine = [], []
    rgb_map, rgb_map_fine = [], []
    acc_map, acc_map_fine = [], []
    flow, flow_fine = [], []
    for i in range(0, rays_o.shape[0], args.chunk_ray):
        # Forward
        with torch.no_grad():
            rays_o_batch = rays_o[i:(i+args.chunk_ray)]
            rays_d_batch = rays_d[i:(i+args.chunk_ray)]
            viewdirs_batch = viewdirs[i:(i+args.chunk_ray)]
            rays = Rays(rays_o_batch, rays_d_batch, viewdirs_batch,
                        args.n_sample_point, args.n_sample_point_fine, near, far, args.perturb)
            ret_dict = nerf_render(rays, cur_time,
                                   point_embedding, view_embedding, time_embedding,
                                   model, model_fine,
                                   fine_sampling=fine_sampling,
                                   zero_canonical=False,
                                   density_noise_std=0.0,
                                   white_bkgd=args.white_bkgd)

            rgb_map_fine.append(ret_dict['rgb_map_fine'])
            acc_map_fine.append(ret_dict['acc_map_fine'])

            # Cast depth to 3D points
            points_fine_batch = rays_o_batch + ret_dict['depth_map_fine'].unsqueeze(1) * rays_d_batch
            if args.use_ndc:
                points_fine_batch = restore_ndc_points(points_fine_batch, img_h, img_w, focal, near_plane=1.)
            points_fine.append(points_fine_batch)

            # Re-query flow for casted points
            _, _, flow_fine_batch = point_query(points_fine_batch, viewdirs_batch, cur_time,
                                                point_embedding, view_embedding, time_embedding,
                                                model, model_fine,
                                                fine_sampling=fine_sampling,
                                                zero_canonical=False)
            flow_fine.append(flow_fine_batch)

    rgb_map_fine = torch.cat(rgb_map_fine, 0)
    points_fine = torch.cat(points_fine, 0)
    acc_map_fine = torch.cat(acc_map_fine, 0)
    flow_fine = torch.cat(flow_fine, 0)

    # Filter out empty points
    acc_thresh = 0.99
    valid_fine = (acc_map_fine > acc_thresh)
    points_fine, rgb_map_fine = points_fine[valid_fine], rgb_map_fine[valid_fine]
    flow_fine = flow_fine[valid_fine]
    viewdirs = viewdirs[valid_fine]

    # Downsample
    fps_idx = fps_downsample(points_fine, n_sample_point=n_sample_point)
    points_fine, rgb_map_fine = points_fine[fps_idx], rgb_map_fine[fps_idx]
    flow_fine = flow_fine[fps_idx]
    viewdirs = viewdirs[fps_idx]

    # Prepare tracking
    points_sequence = [points_fine]
    track_target = points_fine + flow_fine

    """
    Perform tracking in following frames
    """
    track_offset = torch.linspace(-track_box_size, track_box_size, steps=track_box_sample+1)
    track_offset = torch.stack(torch.meshgrid([track_offset, track_offset, track_offset]), -1)
    track_offset = track_offset.reshape(-1, 3)
    n_candidate = track_offset.shape[0]
    viewdirs = viewdirs.unsqueeze(1).expand([n_sample_point, n_candidate, 3])
    viewdirs = viewdirs.reshape(-1, 3)

    track_errors = []
    for cur_time in (track_times[1:]):
        cur_time = cur_time.unsqueeze(0)
        points_last = points_sequence[-1]
        points_search = points_last.unsqueeze(1) + track_offset.unsqueeze(0)

        with torch.no_grad():
            # Query flow for points
            points_search = points_search.reshape(-1, 3)
            _, _, flow = point_query(points_search, viewdirs, cur_time,
                                     point_embedding, view_embedding, time_embedding,
                                     model, model_fine,
                                     fine_sampling=fine_sampling,
                                     zero_canonical=False)
            points_search = points_search.reshape(n_sample_point, n_candidate, 3)
            flow = flow.reshape(n_sample_point, n_candidate, 3)

        points_search_warped = points_search + flow
        track_error = (points_search_warped - track_target.unsqueeze(1)).norm(dim=2)
        track_error, track_idx = torch.min(track_error, dim=1)
        points_tracked = points_search[torch.arange(n_sample_point).long(), track_idx.long()]
        points_sequence.append(points_tracked)
        track_errors.append(track_error.mean().item())

    print('Point tracking error:', track_errors)
    print('Mean point tracking error:', np.mean(track_errors), 'std:', np.std(track_errors))

    points_sequence = torch.stack(points_sequence, 0).cpu().numpy()
    rgb_color = rgb_map_fine.cpu().numpy().clip(0., 1.)


    """
    Visualize 3D point sequence
    """
    import time
    points_sequence[:, :, [1, 2]] = points_sequence[:, :, [2, 1]]
    points_sequence[:, :, 2] *= -1
    save_vis_dir = osp.join(save_render_base, 'pc')
    os.makedirs(save_vis_dir, exist_ok=True)
    for t in range(points_sequence.shape[0]):
        if t == 0:
            pcd = build_colored_pointcloud(points_sequence[t], rgb_color)
            vis = o3d.visualization.Visualizer()
            vis.create_window()
            vis.add_geometry(pcd)
        else:
            pcd_new = build_colored_pointcloud(points_sequence[t], rgb_color)
            pcd.points = pcd_new.points
            pcd.colors = pcd_new.colors
            vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.2)

        save_file = osp.join(save_vis_dir, 'pc_%06d.png'%(t))
        vis.capture_screen_image(save_file)

    """
    Visualize point trajectory
    """
    # save_vis_dir = osp.join(save_render_base, 'flow')
    # os.makedirs(save_vis_dir, exist_ok=True)
    # points_0 = points_sequence[0]
    # for t in range(1, points_sequence.shape[0]):
    #     flow = points_sequence[t] - points_0
    #     save_file = osp.join(save_vis_dir, 'flow_%06d.png'%(t))
    #     visualize_point_flow_plt(points_0, flow, save_file=save_file)