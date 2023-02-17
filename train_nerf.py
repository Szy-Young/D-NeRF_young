import os
import os.path as osp
from tqdm import tqdm
import yaml
import argparse
import numpy as np
import wandb

import torch
import torch.nn as nn

from model import FourierEmbedding, D_NeRF
from camera import Camera, Rays, convert_rays_to_ndc
from render import nerf_render
from metric import mse_to_psnr


# Create tensor on GPU by default ('.to(device)' & '.cuda()' cost time!)
torch.set_default_tensor_type('torch.cuda.FloatTensor')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Config files')
    parser.add_argument('--use_wandb', dest='use_wandb', default=False, action='store_true', help='Use WANDB for logging')

    # Read parameters
    args = parser.parse_args()
    with open(args.config) as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
    for ckey, cvalue in configs.items():
        args.__dict__[ckey] = cvalue

    # Create wandb logger
    if args.use_wandb:
        wandb.init(project=args.wandb['project'],
                   name=args.wandb['name'],
                   config=configs)

    # Fix the random seed
    seed = args.random_seed
    np.random.seed(seed)
    torch.manual_seed(seed)


    # Load the data
    if args.dataset_type == 'blender':
        from data_loader.load_blender import load_blender_data
        imgs_train, poses_train, times_train, img_h, img_w, focal = load_blender_data(data_root=args.data_root,
                                                                            split='train',
                                                                            half_res=args.half_res,
                                                                            white_bkgd=args.white_bkgd)
        n_view_train = imgs_train.shape[0]
        assert times_train[0] == 0. and times_train[-1] == 1., 'Time must start at 0 and end at 1!'

        imgs_val, poses_val, times_val, _, _, _ = load_blender_data(data_root=args.data_root,
                                                            split='val',
                                                            half_res=args.half_res,
                                                            white_bkgd=args.white_bkgd)
        n_view_val = imgs_val.shape[0]

        # Define bounds
        near = 2.
        far = 6.

    else:
        raise ValueError('Not implemented!')


    # Create the Fourier embedding
    point_embedding = FourierEmbedding(n_freq=args.n_freq_point)
    view_embedding = FourierEmbedding(n_freq=args.n_freq_view)
    time_embedding = FourierEmbedding(n_freq=args.n_freq_time, input_dim=1)

    # Create the network (coarse)
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
    grad_vars = list(model.parameters())

    # Create the network (fine)
    fine_sampling = (args.n_sample_point_fine > 0)
    assert fine_sampling or args.train_on_coarse, 'At least train on one of coarse/fine rendering results!'
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
        grad_vars += list(model_fine.parameters())

    # Create the loss & optimizer
    img_loss = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    # Create checkpoint path
    exp_base = args.exp_base
    os.makedirs(exp_base, exist_ok=True)


    # Prepare all rays if "use_batching" in training
    if args.use_batching:
        rays_o_train, rays_d_train = [], []
        for pose in poses_train:
            cam = Camera(img_h, img_w, focal, pose)
            rays_o, rays_d = cam.get_rays_np()
            rays_o_train.append(rays_o)
            rays_d_train.append(rays_d)
        rays_o_train = np.stack(rays_o_train, 0)    # (N, H, W, 3)
        rays_d_train = np.stack(rays_d_train, 0)    # (N, H, W, 3)
        rays_train = np.stack([rays_o_train, rays_d_train, imgs_train], 3)    # (N, H, W, 3(o+d+rgb), 3)
        rays_train = np.reshape(rays_train, [-1, 3, 3])
        print('Shuffle all rays...')
        np.random.shuffle(rays_train)
        i_batch = 0

    """
    Training loop
    """
    tbar = tqdm(total=args.n_iters)
    for it in range(1, args.n_iters+1):
        # Sample rays from all views in each iter
        if args.use_batching:
            rays_batch = rays_train[i_batch:(i_batch+args.n_sample_ray)]
            rays_batch = torch.Tensor(rays_batch)
            rays_o, rays_d, target = rays_batch[:, 0], rays_batch[:, 1], rays_batch[:, 2]

            i_batch += args.n_sample_ray
            if i_batch >= rays_train.shape[0]:
                print('Shuffle all rays after an epoch...')
                np.random.shuffle(rays_train)
                i_batch = 0

        # Sample rays from only 1 view in each iter
        else:
            if it >= args.precrop_iters_time:
                sel_view = np.random.choice(n_view_train)
            else:
                # Only sample from first several views (time steps) at early training stage
                skip_factor = it / float(args.precrop_iters_time) * n_view_train
                max_sample = max(int(skip_factor), 3)
                sel_view = np.random.choice(max_sample)
            target = torch.Tensor(imgs_train[sel_view])
            pose = torch.Tensor(poses_train[sel_view])
            cur_time = torch.Tensor([times_train[sel_view]])

            # Sample rays
            cam = Camera(img_h, img_w, focal, pose)
            if it < args.precrop_iters:
                rays_o, rays_d, select_coords = cam.sample_rays(n_sample_ray=args.n_sample_ray,
                                                                precrop_frac=args.precrop_frac)
            else:
                rays_o, rays_d, select_coords = cam.sample_rays(n_sample_ray=args.n_sample_ray)
            target = target[select_coords[:, 0], select_coords[:, 1]]

        # Forward
        viewdirs = rays_d / rays_d.norm(dim=1, keepdim=True)
        if args.use_ndc:
            rays_o, rays_d = convert_rays_to_ndc(rays_o, rays_d, img_h, img_w, focal, near_plane=1.)
        rays = Rays(rays_o, rays_d, viewdirs, args.n_sample_point, args.n_sample_point_fine, near, far, args.perturb)
        ret_dict = nerf_render(rays, cur_time,
                               point_embedding, view_embedding, time_embedding,
                               model, model_fine,
                               fine_sampling=fine_sampling,
                               zero_canonical=args.zero_canonical,
                               density_noise_std=args.density_noise_std,
                               white_bkgd=args.white_bkgd)

        loss_img = img_loss(ret_dict['rgb_map'], target)
        psnr = mse_to_psnr(loss_img.detach().cpu())
        if fine_sampling:
            loss_img_fine = img_loss(ret_dict['rgb_map_fine'], target)
            psnr_fine = mse_to_psnr(loss_img_fine.detach().cpu())
        else:
            loss_img_fine = torch.zeros()
            psnr_fine = torch.zeros()

        if args.train_on_coarse:
            loss = loss_img + loss_img_fine
        else:
            loss = loss_img_fine

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Add train logs
        train_log = {"train_loss": loss_img.item(), "train_loss_fine": loss_img_fine.item(),
                     "train_psnr": psnr.item(), "train_psnr_fine": psnr_fine.item()}

        # Save model checkpoint
        if it % args.save_freq == 0:
            torch.save(model.state_dict(), osp.join(args.exp_base, 'model_%06d.pth.tar'%(it)))
            if model_fine is not None:
                torch.save(model_fine.state_dict(), osp.join(args.exp_base, 'model_%06d_fine.pth.tar'%(it)))

        # Decay learning rate
        new_lrate = args.lrate * (args.lrate_decay ** (it / args.lrate_decay_step))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        train_log['lrate'] = new_lrate


        # Validation
        if it % args.val_freq == 0:
            # sel_view = np.random.choice(n_view_val)
            sel_view = (it // args.val_freq) % n_view_val
            target = torch.Tensor(imgs_val[sel_view])
            pose = torch.Tensor(poses_val[sel_view])
            cur_time = torch.Tensor([times_train[sel_view]])

            # Get rays for all pixels
            cam = Camera(img_h, img_w, focal, pose)
            rays_o, rays_d = cam.get_rays()
            rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)
            viewdirs = rays_d / rays_d.norm(dim=1, keepdim=True)
            if args.use_ndc:
                rays_o, rays_d = convert_rays_to_ndc(rays_o, rays_d, img_h, img_w, focal, near_plane=1.)

            # Sample several time steps to observe the motion under this view
            sampled_time = torch.linspace(0., 1., steps=5)
            render_times = torch.cat([cur_time, sampled_time])
            rgb_map_logs, rgb_map_fine_logs = [], []

            for tid, render_time in enumerate(render_times):
                render_time = render_time.reshape([1])

                # Batchify
                rgb_map, rgb_map_fine = [], []
                for i in range(0, rays_o.shape[0], args.chunk):
                    # Forward
                    with torch.no_grad():
                        rays_o_batch = rays_o[i:(i+args.chunk)]
                        rays_d_batch = rays_d[i:(i+args.chunk)]
                        viewdirs_batch = viewdirs[i:(i + args.chunk)]
                        rays = Rays(rays_o_batch, rays_d_batch, viewdirs_batch,
                                    args.n_sample_point, args.n_sample_point_fine, near, far, args.perturb)
                        ret_dict = nerf_render(rays, render_time,
                                               point_embedding, view_embedding, time_embedding,
                                               model, model_fine,
                                               fine_sampling=fine_sampling,
                                               zero_canonical=args.zero_canonical,
                                               density_noise_std=0.0,
                                               white_bkgd=args.white_bkgd)

                        rgb_map.append(ret_dict['rgb_map'])
                        if fine_sampling:
                            rgb_map_fine.append(ret_dict['rgb_map_fine'])

                rgb_map = torch.cat(rgb_map, 0).reshape(target.shape)
                if fine_sampling > 0:
                    rgb_map_fine = torch.cat(rgb_map_fine, 0).reshape(target.shape)

                # Add validation logs
                if tid == 0:
                    loss_img = img_loss(rgb_map, target)
                    psnr = mse_to_psnr(loss_img.detach().cpu())
                    if fine_sampling > 0:
                        loss_img_fine = img_loss(rgb_map_fine, target)
                        psnr_fine = mse_to_psnr(loss_img_fine.detach().cpu())
                    else:
                        loss_img_fine = torch.zeros()
                        psnr_fine = torch.zeros()
                    val_log = {"val_loss": loss_img.item(), "val_loss_fine": loss_img_fine.item(),
                               "val_psnr": psnr.item(), "val_psnr_fine": psnr_fine.item()}
                    train_log = train_log | val_log

                # Add rendering visualization logs
                render_time = render_time[0].cpu().numpy()
                rgb_map = rgb_map.cpu().numpy().clip(0., 1.)
                rgb_map = wandb.Image(rgb_map, caption="coarse rendering %f"%(render_time))
                rgb_map_logs.append(rgb_map)
                rgb_map_fine = rgb_map_fine.cpu().numpy().clip(0., 1.)
                rgb_map_fine = wandb.Image(rgb_map_fine, caption="fine rendering %f"%(render_time))
                rgb_map_fine_logs.append(rgb_map_fine)

            render_log = {'val_img': rgb_map_logs, 'val_img_fine': rgb_map_fine_logs}
            train_log = train_log | render_log

        # Logging
        if args.use_wandb:
            wandb.log(train_log)

        tbar.update(1)