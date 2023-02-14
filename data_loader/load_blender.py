import os
import os.path as osp
import torch
import numpy as np
import imageio 
import json
import cv2


def load_blender_data(data_root,
                      split='test',
                      half_res=False,
                      white_bkgd=False,
                      testskip=1,):
    # Load meta info
    with open(osp.join(data_root, 'transforms_%s.json'%(split)), 'r') as f:
        meta = json.load(f)

    # Skip some views during validation/testing
    if split == 'train' or testskip == 0:
        skip = 1
    else:
        skip = testskip

    imgs, poses, times = [], [], []
    for t, frame in enumerate(meta['frames'][::skip]):
        img_file = osp.join(data_root, frame['file_path'] + '.png')
        img = imageio.imread(img_file)
        imgs.append(img)
        pose = np.array(frame['transform_matrix'])
        poses.append(pose)
        cur_time = frame['time'] if 'time' in frame else float(t)/(len(meta['frames'][::skip])-1)
        times.append(cur_time)

    imgs = np.stack(imgs, 0)        # (N, H, W, 4)
    imgs = (imgs / 255.).astype(np.float32)
    poses = np.stack(poses, 0).astype(np.float32)      # (N, 4, 4)
    times = np.array(times).astype(np.float32)      # (N,)

    img_h, img_w = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = 0.5 * img_w / np.tan(0.5 * camera_angle_x)
    
    if half_res:
        img_h = img_h // 2
        img_w = img_w // 2
        focal = focal / 2.

        imgs_half_res = np.zeros((imgs.shape[0], img_h, img_w, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (img_w, img_h), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res

    if white_bkgd:
        imgs = imgs[..., :3] * imgs[..., 3:] + (1. - imgs[..., 3:])
    else:
        imgs = imgs[..., :3]

    return imgs, poses, times, img_h, img_w, focal


if __name__ == '__main__':
    data_root = '/home/ziyang/Desktop/Datasets/nerf_dataset/dnerf/mutant'
    split = 'test'
    half_res = True
    testskip = 20
    white_bkgd = True
    imgs, depths, poses, img_h, img_w, focal = load_blender_data(data_root=data_root,
                                                                 split=split,
                                                                 half_res=half_res,
                                                                 white_bkgd=white_bkgd,
                                                                 testskip=testskip)