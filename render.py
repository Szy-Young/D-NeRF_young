import torch


def volume_render(rgb,
                  density,
                  flow,
                  z_vals,
                  rays_d,
                  white_bkgd=False):
    """
    :param rgb: (Nr, Np, 3) torch.Tensor.
    :param density: (Nr, Np) torch.Tensor.
    :param flow: (Nr, Np, 3) torch.Tensor.
    :param z_vals: (Nr, Np) torch.Tensor.
    :param rays_d: (Nr, 3) torch.Tensor.
    :return:
        rgb_map: (Nr, 3) torch.Tensor.
        depth_map: (Nr,) torch.Tensor.
        acc_map: (Nr,) torch.Tensor.
        disp_map: (Nr,) torch.Tensor.
        weights: (Nr, Np) torch.Tensor.
    """
    dists = z_vals[:, 1:] - z_vals[:, :-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[:, :1].shape)], 1)
    dists = dists * rays_d.norm(dim=1, keepdim=True)    # (Nr, Np)

    alpha = 1. - torch.exp(- density * dists)      # (Nr, Np)
    transparency = torch.cumprod(torch.cat([torch.ones(alpha[:, :1].shape), 1. - alpha + 1e-10], 1), 1)[:, :-1]
    weights = alpha * transparency      # (Nr, Np)

    rgb_map = torch.sum(weights.unsqueeze(2) * rgb, 1)      # (Nr, 3)
    depth_map = torch.sum(weights * z_vals, 1)      # (Nr,)
    acc_map = torch.sum(weights, 1)     # (Nr,)
    disp_map = 1. / (depth_map / acc_map).clamp(min=1e-10)      # (Nr,)
    flow_map = torch.sum(weights.unsqueeze(2) * flow, 1)        # (Nr, 3)

    if white_bkgd:
        rgb_map = rgb_map + (1. - acc_map.unsqueeze(1))

    return rgb_map, depth_map, acc_map, disp_map, flow_map, weights


def nerf_render(rays,
                cur_time,
                point_embedding,
                view_embedding,
                time_embedding,
                model,
                model_fine=None,
                fine_sampling=False,
                zero_canonical=True,
                density_noise_std=0.0,
                white_bkgd=False):
    # Sample points on rays
    points, viewdirs, z_vals = rays.sample_points()
    n_ray, n_point = points.shape[:2]
    points, viewdirs = points.reshape(-1, 3), viewdirs.reshape(-1, 3)
    cur_time = time_embedding(cur_time)

    # Query
    points = point_embedding(points)
    viewdirs = view_embedding(viewdirs)
    # time_shape = [points.shape[0], points.shape[1], cur_time.shape[0]]
    time_shape = [points.shape[0], cur_time.shape[0]]
    time = cur_time.expand(time_shape)
    rgb, density, flow = model(points, viewdirs, time, zero_canonical, density_noise_std)
    rgb, density = rgb.reshape(n_ray, n_point, 3), density.reshape(n_ray, n_point)
    flow = flow.reshape(n_ray, n_point, 3)

    # Differentiable volume render
    z_vals = z_vals
    rays_d = rays.rays_d
    rgb_map, depth_map, acc_map, disp_map, flow_map, weights = volume_render(rgb,
                                                                             density,
                                                                             flow,
                                                                             z_vals,
                                                                             rays_d,
                                                                             white_bkgd)
    # Collect_results
    ret_dict = {'rgb_map': rgb_map,
                'depth_map': depth_map,
                'acc_map': acc_map,
                'disp_map': disp_map,
                'flow_map': flow_map}


    # Query & volume render with model-fine
    if fine_sampling:
        # Re-sample points on rays w.r.t importance (points in occupied regions)
        points, viewdirs, z_vals = rays.sample_points_fine(z_vals,
                                                           weights.detach())
        n_ray, n_point = points.shape[:2]
        points, viewdirs = points.reshape(-1, 3), viewdirs.reshape(-1, 3)

        # Query
        points = point_embedding(points)
        viewdirs = view_embedding(viewdirs)
        # time_shape = [points.shape[0], points.shape[1], cur_time.shape[0]]
        time_shape = [points.shape[0], cur_time.shape[0]]
        time = cur_time.expand(time_shape)
        if model_fine is not None:
            rgb, density, flow = model_fine(points, viewdirs, time, zero_canonical, density_noise_std)
        else:
            rgb, density, flow = model(points, viewdirs, time, zero_canonical, density_noise_std)
        rgb, density = rgb.reshape(n_ray, n_point, 3), density.reshape(n_ray, n_point)
        flow = flow.reshape(n_ray, n_point, 3)

        # Differentiable volume render
        rgb_map, depth_map, acc_map, disp_map, flow_map, weights = volume_render(rgb,
                                                                                 density,
                                                                                 flow,
                                                                                 z_vals,
                                                                                 rays_d,
                                                                                 white_bkgd)
        # Collect_results
        ret_dict_fine = {'rgb_map_fine': rgb_map,
                         'depth_map_fine': depth_map,
                         'acc_map_fine': acc_map,
                         'disp_map_fine': disp_map,
                         'flow_map_fine': flow_map}
        ret_dict = ret_dict | ret_dict_fine

    return ret_dict


def point_query(points,
                viewdirs,
                cur_time,
                point_embedding,
                view_embedding,
                time_embedding,
                model,
                model_fine=None,
                fine_sampling=False,
                zero_canonical=True):
    """
    Direct query values for points, without doing rendering
    :param points: (N, 3).
    :param viewdirs: (N, 3), if provided.
    :param cur_time: (1,)
    :return:
        flow: (N, 3)
    """
    points = point_embedding(points)
    viewdirs = view_embedding(viewdirs)

    cur_time = time_embedding(cur_time)
    time_shape = [points.shape[0], cur_time.shape[0]]
    time = cur_time.expand(time_shape)

    # Query
    if fine_sampling and (model_fine is not None):
        rgb, density, flow = model_fine(points, viewdirs, time, zero_canonical, density_noise_std=0.)
    else:
        rgb, density, flow = model(points, viewdirs, time, zero_canonical, density_noise_std=0.)
    return rgb, density, flow