# This file is part of Uni-SLAM: Uncertainty-Aware Neural Implicit SLAM
# for Real-Time Dense Indoor Scene Reconstruction.
# Project page: https://shaoxiang777.github.io/project/uni-slam/
#
# Copyright 2024 Shaoxiang Wang <shaoxiang.wang@dfki.de>
# Licensed under the Apache License, Version 2.0.
# http://www.apache.org/licenses/LICENSE-2.0
#
# This work builds upon ESLAM (https://github.com/amslabtech/eslam),
# which in turn is based on NICE-SLAM (https://github.com/cvg/nice-slam).
# Both are licensed under the Apache License, Version 2.0.
#
# This file contains modified code originally from ESLAM and NICE-SLAM.
# It is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS
# OF ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

import argparse

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from tqdm import tqdm

import sys
sys.path.append('.')
from src.utils.datasets import get_dataset
from src import config

def cull_mesh(mesh_file, cfg, args, device, eval_rec, estimate_c2w_list=None):
    """
    Cull the mesh by removing the points that are not visible in any of the frames.
    The output mesh file will be saved in the same directory as the input mesh file.
    Args:
        mesh_file (str): path to the mesh file
        cfg (dict): configuration
        args (argparse.Namespace): arguments
        device (torch.device): device
        estimate_c2w_list (list): list of estimated camera poses, if None, it uses the ground truth camera poses
    Returns:
        None

    """
    frame_reader = get_dataset(cfg, args, 1, device=device)

    truncation = cfg['model']['truncation']
    H, W, fx, fy, cx, cy = cfg['cam']['H'], cfg['cam']['W'], cfg['cam']['fx'], cfg['cam']['fy'], cfg['cam']['cx'], cfg['cam']['cy']

    if estimate_c2w_list is not None:
        n_imgs = len(estimate_c2w_list)
    else:
        n_imgs = len(frame_reader)

    mesh = trimesh.load(mesh_file, process=False)
    pc = mesh.vertices

    whole_mask = np.ones(pc.shape[0]).astype('bool')
    for i in tqdm(range(0, n_imgs, 1)):
        _, _, depth, c2w, ray_d = frame_reader[i]
        depth, c2w = depth.to(device), c2w.to(device)

        if not estimate_c2w_list is None:
            c2w = estimate_c2w_list[i].to(device)

        points = pc.copy()
        points = torch.from_numpy(points).to(device)

        w2c = torch.inverse(c2w)
        K = torch.from_numpy(
            np.array([[fx, .0, cx], [.0, fy, cy], [.0, .0, 1.0]]).reshape(3, 3)).to(device)
        ones = torch.ones_like(points[:, 0]).reshape(-1, 1).to(device)
        homo_points = torch.cat(
            [points, ones], dim=1).reshape(-1, 4, 1).to(device).float()
        cam_cord_homo = w2c@homo_points
        cam_cord = cam_cord_homo[:, :3]

        cam_cord[:, 0] *= -1
        uv = K.float()@cam_cord.float()
        z = uv[:, -1:]+1e-5
        uv = uv[:, :2]/z
        uv = uv.squeeze(-1)

        grid = uv[None, None].clone()
        grid[..., 0] = grid[..., 0] / W
        grid[..., 1] = grid[..., 1] / H
        grid = 2 * grid - 1
        depth_samples = F.grid_sample(depth[None, None], grid, padding_mode='zeros', align_corners=True).squeeze()

        edge = 0
        if eval_rec:
            mask = (depth_samples + truncation >= -z[:, 0, 0]) & (0 <= -z[:, 0, 0]) & (uv[:, 0] < W - edge) & (uv[:, 0] > edge) & (uv[:, 1] < H - edge) & (uv[:, 1] > edge)
        else:
            mask = (0 <= -z[:, 0, 0]) & (uv[:, 0] < W - edge) & (uv[:, 0] > edge) & (uv[:, 1] < H-edge) & (uv[:, 1] > edge)

        mask = mask.cpu().numpy()

        whole_mask &= ~mask

    face_mask = whole_mask[mesh.faces].all(axis=1)
    mesh.update_faces(~face_mask)
    mesh.remove_unreferenced_vertices()
    mesh.process(validate=False)

    mesh_ext = mesh_file.split('.')[-1]
    output_file = mesh_file[:-len(mesh_ext) - 1] + '_culled.' + mesh_ext

    mesh.export(output_file)


def cull_out_bound_mesh(mesh, mesh_bound, cfg, args, device, estimate_c2w_list):
    """
    Remove a around the outside of a bound
    Args:
        mesh_file (str): path to the mesh file
        cfg (dict): configuration
        args (argparse.Namespace): arguments
        device (torch.device): device
        estimate_c2w_list (list): list of estimated camera poses, if None, it uses the ground truth camera poses
    Returns:
        None

    """
    frame_reader = get_dataset(cfg, args, 1, device=device)

    H, W, fx, fy, cx, cy = cfg['cam']['H'], cfg['cam']['W'], cfg['cam']['fx'], cfg['cam']['fy'], cfg['cam']['cx'], cfg['cam']['cy']

    if estimate_c2w_list is not None:
        n_imgs = len(estimate_c2w_list)
    else:
        n_imgs = len(frame_reader)

    pc = mesh.vertices

    print('Start to remove bound...')
    pc = mesh.vertices
    bound_mask = []
    for i in tqdm(range(0, n_imgs, 1)):
        total_batches = int(pc.shape[0] // n_imgs)
        bound_mask.append(mesh_bound.contains(pc[i*total_batches:(i+1)*total_batches]))

    bound_mask.append(mesh_bound.contains(pc[(i+1)*total_batches:]))    
    bound_mask = np.concatenate(bound_mask, axis=0)    
    bound_face_mask = bound_mask[mesh.faces].all(axis=1)
    mesh.update_faces(bound_face_mask)
    mesh.remove_unreferenced_vertices()

    return mesh

## It is also possible to use the cull_mesh function in the following way, where the ground truth camera poses are used.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Arguments to cull the mesh.'
    )

    parser.add_argument('config', type=str,  help='path to the config file')
    parser.add_argument('--input_mesh', type=str, help='path to the mesh to be culled')

    args = parser.parse_args()
    args.input_folder = None

    cfg = config.load_config(args.config, 'configs/UNISLAM.yaml')

    cull_mesh(args.input_mesh, cfg, args, 'cuda')