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

import torch
import copy
import os
import time
import json
import time

from colorama import Fore, Style
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.common import (matrix_to_cam_pose, cam_pose_to_matrix, get_samples)
from src.utils.datasets import get_dataset
from src.utils.Frame_Visualizer import Frame_Visualizer
from src.tools.eval_ate import pose_evaluation

class Tracker(object):
    """
    Tracking main class.
    Args:
        cfg (dict): config dict
        args (argparse.Namespace): arguments
        eslam (ESLAM): ESLAM object
    """
    def __init__(self, cfg, args, unislam):
        self.cfg = cfg
        self.args = args

        self.scale = cfg['scale']

        self.idx = unislam.idx
        self.bound = unislam.bound
        self.mesher = unislam.mesher
        self.output = unislam.output
        self.verbose = unislam.verbose
        self.renderer = unislam.renderer
        self.gt_c2w_list = unislam.gt_c2w_list
        self.mapping_idx = unislam.mapping_idx
        self.mapping_cnt = unislam.mapping_cnt
        self.LC_cnt = unislam.LC_cnt
        self.tracking_back = unislam.tracking_back
        self.shared_decoders = unislam.shared_decoders
        self.estimate_c2w_list = unislam.estimate_c2w_list
        self.m_iters = unislam.m_iters
        self.addtional_map_records = unislam.addtional_map_records
        self.truncation = unislam.truncation
        self.tracking_rendered_weight_list = unislam.tracking_rendered_weight_list

        if cfg['grid_mode'] == 'hash_grid':
            self.shared_hash_grids_xyz = unislam.shared_hash_grids_xyz
            self.shared_c_hash_grids_xyz = unislam.shared_c_hash_grids_xyz

        self.cam_lr_T = cfg['tracking']['lr_T']
        self.cam_lr_R = cfg['tracking']['lr_R']
        self.device = cfg['device']
        self.num_cam_iters = cfg['tracking']['iters']
        self.gt_camera = cfg['tracking']['gt_camera']
        self.tracking_pixels = cfg['tracking']['pixels']
        self.vis_pose_freq = cfg['tracking']['vis_pose_freq']
        self.w_sdf_fs = cfg['tracking']['w_sdf_fs']
        self.w_sdf_center = cfg['tracking']['w_sdf_center']
        self.w_sdf_tail = cfg['tracking']['w_sdf_tail']
        self.w_depth = cfg['tracking']['w_depth']
        self.w_color = cfg['tracking']['w_color']
        self.ignore_edge_W = cfg['tracking']['ignore_edge_W']
        self.ignore_edge_H = cfg['tracking']['ignore_edge_H']
        self.const_speed_assumption = cfg['tracking']['const_speed_assumption']
        self.activated_mapping_mode = cfg['tracking']['activated_mapping_mode']
        self.uncertainty_ts = cfg['tracking']['uncertainty_ts']
        self.t_mask_mode = cfg['t_mask_mode']

        self.every_frame = cfg['mapping']['every_frame']
        self.no_vis_on_first_frame = cfg['tracking']['no_vis_on_first_frame']

        self.prev_mapping_idx = -1
        self.frame_reader = get_dataset(cfg, args, self.scale, device=self.device)
        self.n_img = len(self.frame_reader)
        self.freq =cfg['tracking']['vis_freq']
        self.frame_loader = DataLoader(self.frame_reader, batch_size=1, shuffle=False,
                                       num_workers=4, pin_memory=True, prefetch_factor=2)

        self.visualizer = Frame_Visualizer(freq=cfg['tracking']['vis_freq'], inside_freq=cfg['tracking']['vis_inside_freq'],
                                           vis_dir=os.path.join(self.output, 'tracking_vis'), renderer=self.renderer,
                                           truncation=self.truncation, verbose=self.verbose, device=self.device)

        self.H, self.W, self.fx, self.fy, self.cx, self.cy = unislam.H, unislam.W, unislam.fx, unislam.fy, unislam.cx, unislam.cy

        self.decoders = copy.deepcopy(self.shared_decoders)
        if cfg['grid_mode'] == 'hash_grid':
            self.hash_grids_xyz = copy.deepcopy(self.shared_hash_grids_xyz)
            self.c_hash_grids_xyz = copy.deepcopy(self.shared_c_hash_grids_xyz)
        
        for p in self.decoders.parameters():
            p.requires_grad_(False)

    def sdf_losses(self, sdf, z_vals, gt_depth):
        """
        Computes the losses for a signed distance function (SDF) given its values, depth values and ground truth depth.

        Args:
        - self: instance of the class containing this method
        - sdf: a tensor of shape (R, N) representing the SDF values
        - z_vals: a tensor of shape (R, N) representing the depth values
        - gt_depth: a tensor of shape (R,) containing the ground truth depth values

        Returns:
        - sdf_losses: a scalar tensor representing the weighted sum of the free space, center, and tail losses of SDF
        """

        front_mask = torch.where(z_vals < (gt_depth[:, None] - self.truncation),
                                 torch.ones_like(z_vals), torch.zeros_like(z_vals)).bool()

        back_mask = torch.where(z_vals > (gt_depth[:, None] + self.truncation),
                                torch.ones_like(z_vals), torch.zeros_like(z_vals)).bool()

        center_mask = torch.where((z_vals > (gt_depth[:, None] - 0.4 * self.truncation)) *
                                  (z_vals < (gt_depth[:, None] + 0.4 * self.truncation)),
                                  torch.ones_like(z_vals), torch.zeros_like(z_vals)).bool()

        tail_mask = (~front_mask) * (~back_mask) * (~center_mask)

        fs_loss = torch.mean(torch.square(sdf[front_mask] - torch.ones_like(sdf[front_mask])))
        center_loss = torch.mean(torch.square(
            (z_vals + sdf * self.truncation)[center_mask] - gt_depth[:, None].expand(z_vals.shape)[center_mask]))
        tail_loss = torch.mean(torch.square(
            (z_vals + sdf * self.truncation)[tail_mask] - gt_depth[:, None].expand(z_vals.shape)[tail_mask]))

        sdf_losses = self.w_sdf_fs * fs_loss + self.w_sdf_center * center_loss + self.w_sdf_tail * tail_loss

        return sdf_losses

    def optimize_tracking(self, cam_pose, gt_color, gt_depth, batch_size, optimizer):
        """
        Do one iteration of camera tracking. Sample pixels, render depth/color, calculate loss and backpropagation.

        Args:
            cam_pose (tensor): camera pose.
            gt_color (tensor): ground truth color image of the current frame.
            gt_depth (tensor): ground truth depth image of the current frame.
            batch_size (int): batch size, number of sampling rays.
            optimizer (torch.optim): camera optimizer.

        Returns:
            loss (float): The value of loss.
        """

        if self.cfg['grid_mode'] == 'hash_grid':
            scene_rep = (self.hash_grids_xyz, self.c_hash_grids_xyz)
     
        device = self.device
        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy

        c2w = cam_pose_to_matrix(cam_pose)
        batch_rays_o, batch_rays_d, batch_gt_depth_all, batch_gt_color = get_samples(self.ignore_edge_H, H-self.ignore_edge_H,
                                                                                 self.ignore_edge_W, W-self.ignore_edge_W,
                                                                                 batch_size, H, W, fx, fy, cx, cy, c2w,
                                                                                 gt_depth, gt_color, device)

        # should pre-filter those out of bounding box depth value
        with torch.no_grad():
            det_rays_o = batch_rays_o.clone().detach().unsqueeze(-1)  # (N, 3, 1)
            det_rays_d = batch_rays_d.clone().detach().unsqueeze(-1)  # (N, 3, 1)
            t = (self.bound.unsqueeze(0).to(
                device) - det_rays_o) / det_rays_d
            t, _ = torch.min(torch.max(t, dim=2)[0], dim=1)
            inside_mask = t >= batch_gt_depth_all                
            inside_mask = inside_mask & (batch_gt_depth_all > 0)
        
        # # should pre-filter those out of bounding box depth value
        # with torch.no_grad():
        #     det_rays_o = batch_rays_o.clone().detach().unsqueeze(-1)  # (N, 3, 1)
        #     det_rays_d = batch_rays_d.clone().detach().unsqueeze(-1)  # (N, 3, 1)
        #     t = (self.bound.unsqueeze(0).to(
        #         device) - det_rays_o) / det_rays_d
        #     t, _ = torch.min(torch.max(t, dim=2)[0], dim=1)
        #     t = t.unsqueeze(-1).unsqueeze(-1) 
        #     ray_lengths = t * det_rays_d
        #     ray_lengths_norm = torch.norm(ray_lengths, p=2, dim=1).squeeze(-1)  
        #     inside_mask = ray_lengths_norm >= batch_gt_depth                  
        #     inside_mask = inside_mask & (batch_gt_depth > 0)

        batch_rays_d = batch_rays_d[inside_mask]
        batch_rays_o = batch_rays_o[inside_mask]
        batch_gt_depth = batch_gt_depth_all[inside_mask]
        batch_gt_color = batch_gt_color[inside_mask]

        ret = self.renderer.render_batch_ray(scene_rep, self.decoders, batch_rays_d, batch_rays_o,
                                             self.device, self.truncation, gt_depth=batch_gt_depth)
        

        termination_prob, pixel_unc, depth, color, sdf, z_vals, rendered_depth_uncertainty = ret 

        alpha = 1 - pixel_unc.detach()
        alpha_mask = alpha >0.99 

        ## Filtering the rays for which the rendered depth error is greater than 10 times of the median depth error
        depth_error = (batch_gt_depth - depth.detach()).abs()
        error_median = depth_error.median()
        depth_mask = (depth_error < 10 * error_median)   

        depth_mask = depth_mask & alpha_mask
        
        if self.t_mask_mode == "original":
            ## SDF losses
            loss = self.sdf_losses(sdf[depth_mask], z_vals[depth_mask], batch_gt_depth[depth_mask])

            ## Color Loss
            loss = loss + self.w_color * torch.square(batch_gt_color - color)[depth_mask].mean()

            ### Depth loss
            loss = loss + self.w_depth * torch.square(batch_gt_depth[depth_mask] - depth[depth_mask]).mean()

        elif self.t_mask_mode == "no_mask":
            ## SDF losses
            loss = self.sdf_losses(sdf, z_vals, batch_gt_depth)

            ## Color Loss
            loss = loss + (self.w_color * torch.square(batch_gt_color - color)).mean()

            ### Depth loss
            loss = loss + (self.w_depth * torch.square(batch_gt_depth - depth)).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item(), pixel_unc

    def update_params_from_mapping(self):
        """
        Update the parameters of scene representation from the mapping thread.
        """
        if self.mapping_idx[0] != self.prev_mapping_idx:
            if self.verbose:
                print('Tracking: update the parameters from mapping')

            self.decoders.load_state_dict(self.shared_decoders.state_dict())

            if self.cfg['grid_mode'] == 'hash_grid':
                for hash_grids, self_hash_grids in zip(
                    [self.shared_hash_grids_xyz],
                    [self.hash_grids_xyz]):
                    for i, hash_grid in enumerate(hash_grids):
                        self_hash_grids[i] = hash_grid

                for c_hash_grids, self_c_hash_grids in zip(
                        [self.shared_c_hash_grids_xyz],
                        [self.c_hash_grids_xyz]):
                    for i, c_hash_grid in enumerate(c_hash_grids):
                        self_c_hash_grids[i] = c_hash_grid

            self.prev_mapping_idx = self.mapping_idx[0].clone()

    def run(self):
        """
            Runs the tracking thread for the input RGB-D frames.

        """
        device = self.device
        cfg = self.cfg

        if cfg['grid_mode'] == 'hash_grid':
            scene_rep = (self.hash_grids_xyz, self.c_hash_grids_xyz)

        if self.verbose:
            pbar = self.frame_loader
        else:
            pbar = tqdm(self.frame_loader, smoothing=0.05)

        for idx, gt_color, gt_depth, gt_c2w, ray_d in pbar:
            gt_color = gt_color.to(device, non_blocking=True)
            gt_depth = gt_depth.to(device, non_blocking=True)
            gt_c2w = gt_c2w.to(device, non_blocking=True)

            if not self.verbose:
                pbar.set_description(f"Tracking Frame {idx[0]}")
            idx = idx[0]

            # initiate mapping every self.every_frame frames
            if idx > 0 and (idx % self.every_frame == 1 or self.every_frame == 1 or self.tracking_back[0]==1):
                while self.mapping_idx[0] != idx - 1:
                    time.sleep(0.001)
                pre_c2w = self.estimate_c2w_list[idx - 1].unsqueeze(0).to(device)

            self.update_params_from_mapping()

            if self.verbose:
                print(Fore.MAGENTA)
                print("Tracking Frame ",  idx.item())
                print(Style.RESET_ALL)

            if idx == 0 or self.gt_camera:
                c2w = gt_c2w
                if not self.no_vis_on_first_frame:
                    self.visualizer.save_imgs(idx, 0, gt_depth, gt_color, c2w.squeeze(), scene_rep, self.decoders)

            else:
                if self.const_speed_assumption and idx - 2 >= 0:
                    ## Linear prediction for initialization
                    pre_poses = torch.stack([self.estimate_c2w_list[idx - 2], pre_c2w.squeeze(0)], dim=0)
                    pre_poses = matrix_to_cam_pose(pre_poses)       
                    cam_pose = 2 * pre_poses[1:] - pre_poses[0:1]  
                else:
                    ## Initialize with the last known pose
                    cam_pose = matrix_to_cam_pose(pre_c2w)

                T = torch.nn.Parameter(cam_pose[:, -3:].clone())
                R = torch.nn.Parameter(cam_pose[:,:4].clone())
                cam_para_list_T = [T]
                cam_para_list_R = [R]
                optimizer_camera = torch.optim.Adam([{'params': cam_para_list_T, 'lr': self.cam_lr_T, 'betas':(0.5, 0.999)},
                                                     {'params': cam_para_list_R, 'lr': self.cam_lr_R, 'betas':(0.5, 0.999)}])

                current_min_loss = torch.tensor(float('inf')).float().to(device)
                # for cam_iter in range(self.num_cam_iters):
                cam_iter = 0
                while cam_iter < self.num_cam_iters:
                    cam_pose = torch.cat([R, T], -1)

                    if (idx % self.freq == 0) and cam_iter == self.num_cam_iters-1: 
                        self.visualizer.save_imgs(idx, cam_iter, gt_depth, gt_color, cam_pose, scene_rep, self.decoders)

                    if cam_iter == 0:
                        rendered_weights = None

                    start_time = time.time() 
                    loss, rendered_weights = self.optimize_tracking(cam_pose, gt_color, gt_depth, self.tracking_pixels, optimizer_camera) 

                    if loss < current_min_loss:
                        current_min_loss = loss
                        candidate_cam_pose = cam_pose.clone().detach()
                    
                    cam_iter += 1  

                    if idx > 0 and cam_iter == self.num_cam_iters-1:
                        self.tracking_rendered_weight_list[idx] = rendered_weights.detach().mean() 

                        if self.activated_mapping_mode == True and self.tracking_rendered_weight_list[idx] > self.uncertainty_ts:
                            self.num_cam_iters = cfg['tracking']['iters']*2     # TODO
                            self.m_iters[0] = torch.tensor(cfg['mapping']['iters'])*2
                            self.tracking_back[0] = torch.tensor([1])
                            self.addtional_map_records[idx] = torch.tensor([1])          
                        else:
                            self.num_cam_iters = cfg['tracking']['iters']*1
                            self.m_iters[0] = torch.tensor(cfg['mapping']['iters'])*1
                            self.tracking_back[0] = torch.tensor([0])  

                c2w = cam_pose_to_matrix(candidate_cam_pose)

            self.estimate_c2w_list[idx] = c2w.squeeze(0).clone()
            self.gt_c2w_list[idx] = gt_c2w.squeeze(0).clone()
            pre_c2w = c2w.clone()
            self.idx[0] = idx

            if idx == 0:
                pass
            elif idx%self.vis_pose_freq == 0 or idx==self.n_img-1:
                os.makedirs(f'{self.output}/pose_vis', exist_ok=True)
                plot_path = os.path.join(self.output, 'pose_vis', 'pose_{}.png'.format(idx))
                trans_error, results = pose_evaluation(self.gt_c2w_list[:idx+1], self.estimate_c2w_list[:idx+1], 
                                                       self.tracking_rendered_weight_list[:idx+1], plot_path, 
                                                       scale = self.scale, pose_alignment=cfg['tracking']['pose_alignment'])

                if idx == self.n_img - 1:
                    filename = os.path.join(self.output, 'output.txt')
                    with open(filename, 'a') as file:
                        file.write(json.dumps(results) + '\n')
                        file.write(f"normal mapping frames: {self.n_img/self.every_frame}\n")
                        file.write(f"total mapping frames: {self.mapping_cnt}\n")
                        file.write(f"keyframe selection method: {cfg['mapping']['keyframe_selection_method']}\n")
                        file.write(f"total LC: {self.LC_cnt}\n")
                        