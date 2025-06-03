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
import torch.nn as nn
from torch.utils.data import DataLoader

import os
import time

from colorama import Fore, Style

from src.common import (get_samples, get_samples_all, random_select, matrix_to_cam_pose, cam_pose_to_matrix)
from src.utils.datasets import get_dataset, SeqSampler
from src.utils.Frame_Visualizer import Frame_Visualizer
from src.utils.keyframe import KeyFrameDatabase
from src.tools.eval_recon import calc_3d_metric, calc_2d_metric, eval_rendering
from src.tools.cull_mesh import cull_mesh

class Mapper(object):
    """
    Mapping main class.
    Args:
        cfg (dict): config dict
        args (argparse.Namespace): arguments
        eslam (ESLAM): ESLAM object
    """

    def __init__(self, cfg, args, unislam):

        self.cfg = cfg
        self.args = args

        self.idx = unislam.idx
        self.truncation = unislam.truncation
        self.bound = unislam.bound
        self.logger = unislam.logger
        self.mesher = unislam.mesher
        self.output = unislam.output
        self.verbose = unislam.verbose              
        self.renderer = unislam.renderer
        self.mapping_idx = unislam.mapping_idx
        self.mapping_cnt = unislam.mapping_cnt
        self.LC_cnt = unislam.LC_cnt
        self.tracking_back = unislam.tracking_back
        self.decoders = unislam.shared_decoders
        self.m_iters = unislam.m_iters

        if cfg['grid_mode'] == 'hash_grid':
            self.hash_grids_xyz = unislam.shared_hash_grids_xyz
            self.c_hash_grids_xyz = unislam.shared_c_hash_grids_xyz   

        self.estimate_c2w_list = unislam.estimate_c2w_list
        self.mapping_first_frame = unislam.mapping_first_frame

        self.scale = cfg['scale']
        self.device = cfg['device']
        self.keyframe_device = cfg['keyframe_device']

        self.eval_rec = cfg['meshing']['eval_rec']
        self.joint_opt = False  # Even if joint_opt is enabled, it starts only when there are at least 4 keyframes
        self.joint_opt_cam_lr = cfg['mapping']['joint_opt_cam_lr'] # The learning rate for camera poses during mapping
        self.mesh_freq = cfg['mapping']['mesh_freq']
        self.ckpt_freq = cfg['mapping']['ckpt_freq']
        self.mapping_pixels = cfg['mapping']['pixels']
        self.every_frame = cfg['mapping']['every_frame']
        self.LC_ts  = cfg['mapping']['LC_ts']
        self.LC = cfg['mapping']['LC']
        self.w_sdf_fs = cfg['mapping']['w_sdf_fs']
        self.w_sdf_center = cfg['mapping']['w_sdf_center']
        self.w_sdf_tail = cfg['mapping']['w_sdf_tail']
        self.w_depth = cfg['mapping']['w_depth']
        self.w_color = cfg['mapping']['w_color']
        self.keyframe_every = cfg['mapping']['keyframe_every']
        self.mapping_window_size = cfg['mapping']['mapping_window_size']
        self.no_vis_on_first_frame = cfg['mapping']['no_vis_on_first_frame']
        self.no_log_on_first_frame = cfg['mapping']['no_log_on_first_frame']
        self.no_mesh_on_first_frame = cfg['mapping']['no_mesh_on_first_frame']
        self.keyframe_selection_method = cfg['mapping']['keyframe_selection_method']
        self.m_mask_mode = cfg['m_mask_mode']
        self.addtional_map_records = unislam.addtional_map_records
        self.activated_mapping_mode = cfg['tracking']['activated_mapping_mode']

        self.keyframe_dict = []
        self.keyframe_list = []
        self.frame_reader = get_dataset(cfg, args, self.scale, device=self.device)
        self.n_img = len(self.frame_reader)
        self.frame_loader = DataLoader(self.frame_reader, batch_size=1, num_workers=4, pin_memory=True,
                                       prefetch_factor=2, sampler=SeqSampler(self.n_img, self.every_frame))

        self.visualizer = Frame_Visualizer(freq=cfg['mapping']['vis_freq'], inside_freq=cfg['mapping']['vis_inside_freq'],
                                           vis_dir=os.path.join(self.output, 'mapping_vis'), renderer=self.renderer,
                                           truncation=self.truncation, verbose=self.verbose, device=self.device)

        self.H, self.W, self.fx, self.fy, self.cx, self.cy = unislam.H, unislam.W, unislam.fx, unislam.fy, unislam.cx, unislam.cy
    
    def create_optimizer(self, cfg, lr_factor):

        decoders_para_list = []
        decoders_para_list += list(self.decoders.parameters())    

        if cfg['grid_mode'] == 'hash_grid':
            hash_grids_para = []
            hash_grids_para.append(self.hash_grids_xyz[0].params)

            c_hash_grids_para = []
            c_hash_grids_para.append(self.c_hash_grids_xyz[0].params)
        
        if cfg['grid_mode'] == 'hash_grid':
            optimizer_config = [{'params': decoders_para_list, 'lr': cfg['mapping']['lr']['decoders_lr'] * lr_factor},
                                {'params': hash_grids_para[0], 'lr': cfg['mapping']['lr']['hash_grids_lr'] * lr_factor},
                                {'params': c_hash_grids_para[0],  'lr': cfg['mapping']['lr']['c_hash_grids_lr'] * lr_factor}] 
        
        # model_params_bytes = 0
        # for param_group in optimizer_config:
        #     params = param_group['params']
        #     param_group_bytes = sum(p.numel() * p.element_size() for p in params)
        #     model_params_bytes += param_group_bytes

        # model_params_MB = model_params_bytes / (1024 * 1024)

        # all_params = sum(p.numel() for p in decoders_para_list)+ sum(p.numel() for p in hash_grids_para[0]) \
        #     + sum(tensor.numel() for tensor in c_hash_grids_para[0])
        
        return optimizer_config

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
    
    def keyframe_selection_LC(self, num, idx, gt_color, gt_depth, c2w, num_keyframes, num_samples=8, num_rays=50):
        """
        Select overlapping keyframes to the current camera observation.

        Args:
            num: Count of keyframes in the current list, excluding the last two entries
            idx: frame of current idx
            gt_color: ground truth color image of the current frame.
            gt_depth: ground truth depth image of the current frame.
            c2w: camera to world matrix for target view (3x4 or 4x4 both fine).
            num_keyframes (int): number of overlapping keyframes to select.
            num_samples (int, optional): number of samples/points per ray. Defaults to 8.
            num_rays (int, optional): number of pixels to sparsely sample
                from each image. Defaults to 50.
        Returns:
            selected_keyframe_list (list): list of selected keyframe id.
        """
        device = self.device
        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy

        rays_o, rays_d, gt_depth, gt_color = get_samples(
            0, H, 0, W, num_rays, H, W, fx, fy, cx, cy,
            c2w.unsqueeze(0), gt_depth.unsqueeze(0),gt_color.unsqueeze(0), device)

        gt_depth = gt_depth.reshape(-1, 1)
        nonzero_depth = gt_depth[:, 0] > 0
        rays_o = rays_o[nonzero_depth]
        rays_d = rays_d[nonzero_depth]
        gt_depth = gt_depth[nonzero_depth]
        gt_depth = gt_depth.repeat(1, num_samples)
        t_vals = torch.linspace(0., 1., steps=num_samples).to(device)
        near = gt_depth*0.8
        far = gt_depth+0.5
        z_vals = near * (1.-t_vals) + far * (t_vals)    
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [num_rays, num_samples, 3]
        pts = pts.reshape(1, -1, 3)

        keyframes_c2ws = torch.stack([self.estimate_c2w_list[idx] for idx in self.keyframe_list], dim=0) 
        w2cs = torch.inverse(keyframes_c2ws[:-2])     ## The last two keyframes are already included

        ones = torch.ones_like(pts[..., 0], device=device).reshape(1, -1, 1)
        homo_pts = torch.cat([pts, ones], dim=-1).reshape(1, -1, 4, 1).expand(w2cs.shape[0], -1, -1, -1)
        w2cs_exp = w2cs.unsqueeze(1).expand(-1, homo_pts.shape[1], -1, -1)
        cam_cords_homo = w2cs_exp @ homo_pts
        cam_cords = cam_cords_homo[:, :, :3]
        K = torch.tensor([[fx, .0, cx], [.0, fy, cy],
                          [.0, .0, 1.0]], device=device).reshape(3, 3)
        cam_cords[:, :, 0] *= -1
        uv = K @ cam_cords
        z = uv[:, :, -1:] + 1e-5
        uv = uv[:, :, :2] / z
        edge = 20
        mask = (uv[:, :, 0] < W - edge) * (uv[:, :, 0] > edge) * \
               (uv[:, :, 1] < H - edge) * (uv[:, :, 1] > edge)
        mask = mask & (z[:, :, 0] < 0)
        mask = mask.squeeze(-1)
        percent_inside = mask.sum(dim=1) / uv.shape[1]   # Compute overlap with all previous keyframes except the last two in keyframe_dict.


        if percent_inside.shape[0]>0:
            idx2 = idx                                              
            idx1 = self.keyframe_list[torch.argmax(percent_inside)]  

            if (percent_inside.max() > 0.95 and (idx2 - idx1) > 100) and self.LC == True:
                lc_string = "Loop closure occurred between keyframes with indices {} and {}. Their co-visibility is {}.".format(idx1, idx2, percent_inside.max())
                print(lc_string)

                all_keyframes = list(range(0, num))
                selected_keyframes = all_keyframes[torch.argmax(percent_inside): ]     # Optimize only the frames involved in the current loop closure

                self.LC_cnt[0] += 1
            else:
                selected_keyframes = list(range(0, num))
        else:
            selected_keyframes = list(range(0, num))

        # Local BA performed during tracking back.
        if self.tracking_back[0] == torch.tensor([1]) and self.activated_mapping_mode == True:
            # Using random-overlap selection strategy
            selected_keyframes = torch.nonzero(percent_inside).squeeze(-1)  # returns a 2-D tensor where each row is the index for a nonzero value.
            rnd_inds = torch.randperm(selected_keyframes.shape[0])
            selected_keyframes = selected_keyframes[rnd_inds[:num_keyframes]]

            selected_keyframes = list(selected_keyframes.cpu().numpy())

            # Selecting based on the frame with the highest overlap
            list_keyframe = []
            if percent_inside.shape[0]>0:
                for keyframeid in range(len(percent_inside)):
                    list_keyframe.append({'id': keyframeid, 'percent_inside': percent_inside[keyframeid]})
                list_keyframe = sorted(
                    list_keyframe, key=lambda i: i['percent_inside'], reverse=True)   # Sort the overlapping values in descending order.
                selected_keyframe_list = [dic['id']
                                        for dic in list_keyframe if dic['percent_inside'] > 0.00] 
                selected_keyframes = selected_keyframe_list[:num_keyframes]   

        return selected_keyframes


    def optimize_mapping(self, iters, lr_factor, idx, cur_gt_color, cur_gt_depth, gt_cur_c2w, keyframe_dict, keyframe_list, cur_c2w, cur_rays_d):
        """
        Mapping iterations. Sample pixels from selected keyframes,
        then optimize scene representation and camera poses(if joint_opt enables).

        Args:
            iters (int): number of mapping iterations.
            lr_factor (float): the factor to times on current lr.
            idx (int): the index of current frame
            cur_gt_color (tensor): gt_color image of the current camera.
            cur_gt_depth (tensor): gt_depth image of the current camera.
            gt_cur_c2w (tensor): groundtruth camera to world matrix corresponding to current frame.
            keyframe_dict (list): a list of dictionaries of keyframes info.
            keyframe_list (list): list of keyframes indices.
            cur_c2w (tensor): the estimated camera to world matrix of current frame. 

        Returns:
            cur_c2w: return the updated cur_c2w, return the same input cur_c2w if no joint_opt
        """

        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy
        cfg = self.cfg
        device = self.device

        if cfg['grid_mode'] == 'hash_grid':
            scene_rep = (self.hash_grids_xyz, self.c_hash_grids_xyz)   

        if len(keyframe_dict) == 0:
            optimize_frame = []
        else:
            if self.keyframe_selection_method == 'global':
                optimize_frame = self.keyframe_selection_LC(len(self.keyframe_dict)-2, idx, cur_gt_color, cur_gt_depth, cur_c2w, self.mapping_window_size-1)
            
        # add the last two keyframes and the current frame(use -1 to denote)
        if len(keyframe_list) > 1:
            optimize_frame = optimize_frame + [len(keyframe_list)-1] + [len(keyframe_list)-2]
            optimize_frame = sorted(optimize_frame)
        optimize_frame += [-1]  ## -1 represents the current frame

        pixs_per_image = self.mapping_pixels//len(optimize_frame)

        gt_depths = []
        gt_colors = []
        c2ws = []
        gt_c2ws = []
        rays_d_list = []
        for frame in optimize_frame:
            # the oldest frame should be fixed to avoid drifting
            if frame != -1:
                gt_depths.append(keyframe_dict[frame]['depth'].to(device))
                gt_colors.append(keyframe_dict[frame]['color'].to(device))
                c2ws.append(keyframe_dict[frame]['est_c2w'])
                gt_c2ws.append(keyframe_dict[frame]['gt_c2w'])
                if self.keyframe_selection_method == "global":
                    rays_d_list.append(keyframe_dict[frame]['rays_d'].to(device)) 
            else:
                if self.keyframe_selection_method == "global":
                    total_pixels = cur_gt_color.shape[0] * cur_gt_color.shape[1]
                    num_pixels_to_save = int(cur_gt_color.shape[0] * cur_gt_color.shape[1]*0.1)  
                    indices = torch.randperm(total_pixels)[:num_pixels_to_save]

                    cur_gt_color_samp = cur_gt_color.reshape(-1, cur_gt_color.shape[2])[indices]
                    cur_gt_depth_samp = cur_gt_depth.reshape(-1)[indices] 
                    cur_rays_d_samp = cur_rays_d.reshape(-1, cur_rays_d.shape[2])[indices]

                    gt_depths.append(cur_gt_depth_samp)
                    gt_colors.append(cur_gt_color_samp)
                    rays_d_list.append(cur_rays_d_samp)
                    
                else:
                    gt_depths.append(cur_gt_depth)
                    gt_colors.append(cur_gt_color)

                c2ws.append(cur_c2w)
                gt_c2ws.append(gt_cur_c2w)

        gt_depths = torch.stack(gt_depths, dim=0)             
        gt_colors = torch.stack(gt_colors, dim=0)                
        c2ws = torch.stack(c2ws, dim=0)
        if self.keyframe_selection_method=="global":
            rays_d_list = torch.stack(rays_d_list, dim=0)           

        optimizer_config = self.create_optimizer(cfg, lr_factor)
        if self.joint_opt:
            cam_poses = nn.Parameter(matrix_to_cam_pose(c2ws[1:]))    
            optimizer_config.append({'params': [cam_poses], 'lr': self.joint_opt_cam_lr})

        # The corresponding lr will be set according to which stage the optimization is in
        self.optimizer = torch.optim.Adam(optimizer_config)  
        
        for joint_iter in range(iters):
            start_time = time.time() 

            if (not (idx == 0 and self.no_vis_on_first_frame)) and joint_iter==iters-1:
                self.visualizer.save_imgs(idx, joint_iter, cur_gt_depth, cur_gt_color, cur_c2w, scene_rep, self.decoders)
            
            if self.joint_opt:
                ## We fix the oldest c2w to avoid drifting
                c2ws_ = torch.cat([c2ws[0:1], cam_pose_to_matrix(cam_poses)], dim=0)
            else:
                c2ws_ = c2ws

            if self.keyframe_selection_method=="global":
                batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color = get_samples_all(
                0, H, 0, W, pixs_per_image, H, W, fx, fy, cx, cy, c2ws_, gt_depths, gt_colors, device, rays_d_list)

                if self.tracking_back[0] == torch.tensor([1]): 
                    pass
                else: 
                    if len(self.keyframe_list) > 20: 
                        cur_b_rays_o, cur_b_rays_d, cur_b_gt_depth, cur_b_gt_color = get_samples_all(
                            0, H, 0, W, 200, H, W, fx, fy, cx, cy, 
                            c2ws_[-10:], gt_depths[-10:], gt_colors[-10:], device, rays_d_list[-10:])
                    
                        batch_rays_o = torch.cat([batch_rays_o, cur_b_rays_o], dim=0) 
                        batch_rays_d = torch.cat([batch_rays_d, cur_b_rays_d], dim=0) 
                        batch_gt_depth = torch.cat([batch_gt_depth, cur_b_gt_depth], dim=0) 
                        batch_gt_color = torch.cat([batch_gt_color, cur_b_gt_color], dim=0)  

            # should pre-filter those out of bounding box depth value
            with torch.no_grad():
                det_rays_o = batch_rays_o.clone().detach().unsqueeze(-1)   
                det_rays_d = batch_rays_d.clone().detach().unsqueeze(-1)
                t = (self.bound.unsqueeze(0).to(
                    device)-det_rays_o)/det_rays_d
                t, _ = torch.min(torch.max(t, dim=2)[0], dim=1)
                inside_mask = t >= batch_gt_depth
            batch_rays_d = batch_rays_d[inside_mask]
            batch_rays_o = batch_rays_o[inside_mask]
            batch_gt_depth = batch_gt_depth[inside_mask]
            batch_gt_color = batch_gt_color[inside_mask]

        
            ret = self.renderer.render_batch_ray(scene_rep, self.decoders,batch_rays_d,
                                                batch_rays_o, device, self.truncation, gt_depth=batch_gt_depth)
            
            termination_prob, pixel_unc, depth, color, sdf, z_vals, rendered_depth_uncertainty = ret 
            
            # pixel_unc: 0 = ray reached on surface, 1 = not reached (high uncertainty)
            alpha = 1 - pixel_unc.detach()
            alpha_mask = alpha >0.99 
            
            depth_mask = (batch_gt_depth > 0)

            depth_mask = depth_mask & alpha_mask

            if self.m_mask_mode == "original":
                ## SDF losses
                loss = self.sdf_losses(sdf[depth_mask], z_vals[depth_mask], batch_gt_depth[depth_mask])

                ## Color loss
                loss = loss + self.w_color * torch.square(batch_gt_color - color).mean()

                ### Depth loss
                loss = loss + self.w_depth * torch.square(batch_gt_depth[depth_mask] - depth[depth_mask]).mean()

            elif self.m_mask_mode == "no_mask":
                ## SDF losse
                loss = self.sdf_losses(sdf, z_vals, batch_gt_depth)

                ## Color loss
                loss = loss + self.w_color * torch.square(batch_gt_color - color).mean()

                ### Depth loss
                loss = loss + self.w_depth * torch.square(batch_gt_depth - depth).mean()


            self.optimizer.zero_grad()
            loss.backward(retain_graph=False)
            self.optimizer.step()

        if self.joint_opt:
            # put the updated camera poses back
            optimized_c2ws = cam_pose_to_matrix(cam_poses.detach())

            camera_tensor_id = 0
            for frame in optimize_frame[1:]:
                if frame != -1:
                    keyframe_dict[frame]['est_c2w'] = optimized_c2ws[camera_tensor_id]
                    camera_tensor_id += 1
                else:
                    cur_c2w = optimized_c2ws[-1]

        return cur_c2w

    def run(self):
        def run(self):
            """
            Runs the mapping thread for the input RGB-D frames.

            Args:
                None

            Returns:
                None
        """
        cfg = self.cfg
        if cfg['grid_mode'] == 'hash_grid':
            scene_rep = (self.hash_grids_xyz, self.c_hash_grids_xyz)  

        idx, gt_color, gt_depth, gt_c2w, rays_d = self.frame_reader[0]
        # data_iterator = iter(self.frame_loader)

        ## Fixing the first camera pose
        self.estimate_c2w_list[0] = gt_c2w

        init_phase = True
        prev_idx = -1
        while True:
            while True:
                idx = self.idx[0].clone()
                if idx == self.n_img-1: ## Last input frame
                    break

                if (idx % self.every_frame == 0 or self.tracking_back[0]==1) and idx != prev_idx:
                    break

                time.sleep(0.001)

            prev_idx = idx

            if self.verbose:
                print(Fore.GREEN)
                print("Mapping Frame ", idx.item())
                print(Style.RESET_ALL)
                print(self.m_iters)

            # _, gt_color, gt_depth, gt_c2w = next(data_iterator)
            _, gt_color, gt_depth, gt_c2w, rays_d = self.frame_reader[idx]
            gt_color = gt_color.squeeze(0).to(self.device, non_blocking=True)
            gt_depth = gt_depth.squeeze(0).to(self.device, non_blocking=True)
            gt_c2w = gt_c2w.squeeze(0).to(self.device, non_blocking=True)
            rays_d = rays_d.squeeze(0).to(self.device, non_blocking=True)

            cur_c2w = self.estimate_c2w_list[idx]

            if not init_phase:
                lr_factor = cfg['mapping']['lr_factor']
            else:
                lr_factor = cfg['mapping']['lr_first_factor']
                self.m_iters[0] = cfg['mapping']['iters_first']

            ## Deciding if camera poses should be jointly optimized
            self.joint_opt = (len(self.keyframe_list) > 4) and cfg['mapping']['joint_opt']

            cur_c2w = self.optimize_mapping(self.m_iters, lr_factor, idx, gt_color, gt_depth, gt_c2w,
                                            self.keyframe_dict, self.keyframe_list, cur_c2w, rays_d)

            if self.joint_opt:
                self.estimate_c2w_list[idx] = cur_c2w

            # add new frame to keyframe set
            if idx % self.keyframe_every == 0  or self.tracking_back[0]==1:
                self.keyframe_list.append(idx)

                if self.keyframe_selection_method == "global":
                    total_pixels = gt_color.shape[0] * gt_color.shape[1]
                    num_pixels_to_save = int(gt_color.shape[0] * gt_color.shape[1]*0.1)  
                    indices = torch.randperm(total_pixels)[:num_pixels_to_save]

                    gt_color = gt_color.reshape(-1, gt_color.shape[2])[indices]
                    gt_depth = gt_depth.reshape(-1)[indices] 
                    rays_d = rays_d.reshape(-1, rays_d.shape[2])[indices]

                    self.keyframe_dict.append({'gt_c2w': gt_c2w, 'idx': idx, 'color': gt_color.to(self.keyframe_device),
                                           'depth': gt_depth.to(self.keyframe_device), 'est_c2w': cur_c2w.clone(), 'rays_d':rays_d.to(self.keyframe_device)})
           

            init_phase = False
            self.mapping_first_frame[0] = 1     # mapping of first frame is done, can begin tracking

            if ((not (idx == 0 and self.no_log_on_first_frame)) and idx % self.ckpt_freq == 0) or idx == self.n_img-1:
                self.logger.log(idx, self.keyframe_list)

            self.mapping_idx[0] = idx
            self.mapping_cnt[0] += 1

            if (idx % self.mesh_freq == 0) and (not (idx == 0 and self.no_mesh_on_first_frame)):
                mesh_out_file = f'{self.output}/mesh/{idx:05d}_mesh.ply'
                self.mesher.get_mesh(mesh_out_file, scene_rep, self.decoders, self.keyframe_dict, idx, self.device)
                cull_mesh(mesh_out_file, self.cfg, self.args, self.device, self.cfg['meshing']['eval_rec'], estimate_c2w_list=self.estimate_c2w_list[:idx+1])

            if idx == self.n_img-1:
                self.eval_img = True
                if self.eval_img:
                    eval_rendering(self.cfg, self.n_img, self.frame_reader,self.estimate_c2w_list,
                                   self.renderer, scene_rep, self.decoders, self.truncation, self.output, self.device)
                    
                if self.eval_rec:
                    mesh_out_file = f'{self.output}/mesh/final_mesh_eval_rec.ply'
                else:
                    mesh_out_file = f'{self.output}/mesh/final_mesh.ply'

                self.mesher.get_mesh(mesh_out_file, scene_rep, self.decoders, self.keyframe_dict, idx, self.device)
                cull_mesh(mesh_out_file, self.cfg, self.args, self.device, self.cfg['meshing']['eval_rec'], estimate_c2w_list=self.estimate_c2w_list)

                break

            if idx == self.n_img-1:
                break