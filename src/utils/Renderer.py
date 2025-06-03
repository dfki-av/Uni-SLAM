# This file is part of Uni-SLAM: Uncertainty-Aware Neural Implicit SLAM
# for Real-Time Dense Indoor Scene Reconstruction.
# Project page: https://shaoxiang777.github.io/project/uni-slam/
#
# Copyright 2024 Shaoxiang Wang <shaoxiang.wang@dfki.de>
# Licensed under the Apache License, Version 2.0.
# http://www.apache.org/licenses/LICENSE-2.0
#
# This work builds upon ESLAM (https://www.idiap.ch/paper/eslam/),
# which in turn is based on NICE-SLAM (https://github.com/cvg/nice-slam).
# Both are licensed under the Apache License, Version 2.0.
#
# This file contains modified code originally from ESLAM and NICE-SLAM.
# It is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS
# OF ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

import torch
from src.common import get_rays, sample_pdf, normalize_3d_coordinate

class Renderer(object):
    """
    Renderer class for rendering depth and color.
    Args:
        cfg (dict): configuration.
        unislam (UNISLAM): UNISLAM object.
        ray_batch_size (int): batch size for sampling rays.
    """
    def __init__(self, cfg, unislam, ray_batch_size=10000):
        self.ray_batch_size = ray_batch_size
        self.cfg = cfg

        self.perturb = cfg['rendering']['perturb']
        self.n_stratified = cfg['rendering']['n_stratified']
        self.n_importance = cfg['rendering']['n_importance']

        self.scale = cfg['scale']
        self.bound = unislam.bound.to(unislam.device, non_blocking=True)

        self.H, self.W, self.fx, self.fy, self.cx, self.cy = unislam.H, unislam.W, unislam.fx, unislam.fy, unislam.cx, unislam.cy

    def perturbation(self, z_vals):
        """
        Add perturbation to sampled depth values on the rays.
        Args:
            z_vals (tensor): sampled depth values on the rays.
        Returns:
            z_vals (tensor): perturbed depth values on the rays.
        """
        # get intervals between samples
        mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape, device=z_vals.device)

        return lower + (upper - lower) * t_rand

    def render_batch_ray(self, scene_rep, decoders, rays_d, rays_o, device, truncation, gt_depth=None):
        """
        Render depth and color for a batch of rays.
        Args:
            scene_rep (Tuple): all feature grids.
            decoders (torch.nn.Module): decoders for TSDF and color.
            rays_d (tensor): ray directions.
            rays_o (tensor): ray origins.
            device (torch.device): device to run on.
            truncation (float): truncation threshold.
            gt_depth (tensor): ground truth depth.
        Returns:
            depth_map (tensor): depth map.
            color_map (tensor): color map.
            volume_densities (tensor): volume densities for sampled points.
            z_vals (tensor): sampled depth values on the rays.

        """
        n_stratified = self.n_stratified
        n_importance = self.n_importance
        n_rays = rays_o.shape[0]

        z_vals = torch.empty([n_rays, n_stratified + n_importance], device=device)
        near = 0.0
        t_vals_uni = torch.linspace(0., 1., steps=n_stratified, device=device)
        t_vals_surface = torch.linspace(0., 1., steps=n_importance, device=device)

        ### pixels with gt depth:
        gt_depth = gt_depth.reshape(-1, 1)
        gt_mask = (gt_depth > 0).squeeze()
        gt_nonezero = gt_depth[gt_mask]

        ## Sampling points around the gt depth (surface)
        gt_depth_surface = gt_nonezero.expand(-1, n_importance)
        z_vals_surface = gt_depth_surface - (1.5 * truncation)  + (3 * truncation * t_vals_surface)

        gt_depth_free = gt_nonezero.expand(-1, n_stratified)
        z_vals_free = near + 1.2 * gt_depth_free * t_vals_uni

        z_vals_nonzero, _ = torch.sort(torch.cat([z_vals_free, z_vals_surface], dim=-1), dim=-1)
        if self.perturb:
            z_vals_nonzero = self.perturbation(z_vals_nonzero)
        z_vals[gt_mask] = z_vals_nonzero

        ### pixels without gt depth (importance sampling):
        if not gt_mask.all():
            with torch.no_grad():
                rays_o_uni = rays_o[~gt_mask].detach()
                rays_d_uni = rays_d[~gt_mask].detach()
                det_rays_o = rays_o_uni.unsqueeze(-1)  # (N, 3, 1)
                det_rays_d = rays_d_uni.unsqueeze(-1)  # (N, 3, 1)
                t = (self.bound.unsqueeze(0) - det_rays_o)/det_rays_d  # (N, 3, 2)
                far_bb, _ = torch.min(torch.max(t, dim=2)[0], dim=1)
                far_bb = far_bb.unsqueeze(-1)
                far_bb += 0.01

                z_vals_uni = near * (1. - t_vals_uni) + far_bb * t_vals_uni
                if self.perturb:
                    z_vals_uni = self.perturbation(z_vals_uni)
                pts_uni = rays_o_uni.unsqueeze(1) + rays_d_uni.unsqueeze(1) * z_vals_uni.unsqueeze(-1)  # [n_rays, n_stratified, 3]

                pts_uni_nor = normalize_3d_coordinate(pts_uni.clone(), self.bound)
                sdf_uni = decoders.get_raw_sdf(pts_uni_nor, scene_rep)
                sdf_uni = sdf_uni.reshape(*pts_uni.shape[0:2])
                alpha_uni = self.sdf2alpha(sdf_uni, decoders.beta)
                weights_uni = alpha_uni * torch.cumprod(torch.cat([torch.ones((alpha_uni.shape[0], 1), device=device)
                                                        , (1. - alpha_uni + 1e-10)], -1), -1)[:, :-1]

                z_vals_uni_mid = .5 * (z_vals_uni[..., 1:] + z_vals_uni[..., :-1])
                z_samples_uni = sample_pdf(z_vals_uni_mid, weights_uni[..., 1:-1], n_importance, det=False, device=device)
                z_vals_uni, ind = torch.sort(torch.cat([z_vals_uni, z_samples_uni], -1), -1)
                z_vals[~gt_mask] = z_vals_uni

        pts = rays_o[..., None, :] + rays_d[..., None, :] * \
              z_vals[..., :, None]  # [n_rays, n_stratified+n_importance, 3]
        
        # Normalize the input to [0, 1] (TCNN convention)
        if self.cfg['grid_mode']== 'hash_grid':
            pts = (pts - self.bound[:, 0]) / (self.bound[:, 1] - self.bound[:, 0])

        raw = decoders(pts, scene_rep)
        alpha = self.sdf2alpha(raw[..., 3], decoders.beta)
        weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1), device=device)
                                                , (1. - alpha + 1e-10)], -1), -1)[:, :-1]

        rendered_rgb = torch.sum(weights[..., None] * raw[..., :3], -2)
        rendered_depth = torch.sum(weights * z_vals, -1)
        
        termination_prob = torch.sum(weights, -1)
        pixel_unc = torch.square(1-torch.sum(weights, -1))   # Density-Aware SLAM 公式（8）  这里的结果是epistemic uncertainty
        # TODO  尝试一下absolute value 以及cubic的和理想
        rendered_depth_uncertainty = torch.sqrt(torch.sum(weights * (rendered_depth[..., None] - z_vals)**2, -1))  # Uncle-Slam 公式（5）

        return termination_prob, pixel_unc, rendered_depth, rendered_rgb, raw[..., 3], z_vals, rendered_depth_uncertainty
        
    def sdf2alpha(self, sdf, beta=10):
        """

        """
        return 1. - torch.exp(-beta * torch.sigmoid(-sdf * beta))

    def render_img(self, scene_rep, decoders, c2w, truncation, device, gt_depth=None):
        
        """
        Renders out depth and color images.
        Args:
            scene_rep (Tuple): feature grids
            decoders (torch.nn.Module): decoders for TSDF and color.
            c2w (tensor, 4*4): camera pose.
            truncation (float): truncation distance.
            device (torch.device): device to run on.
            gt_depth (tensor, H*W): ground truth depth image.
        Returns:
            rendered_depth (tensor, H*W): rendered depth image.
            rendered_rgb (tensor, H*W*3): rendered color image.

        """
        with torch.no_grad():
            H = self.H
            W = self.W
            rays_o, rays_d = get_rays(H, W, self.fx, self.fy, self.cx, self.cy,  c2w, device)
            rays_o = rays_o.reshape(-1, 3)
            rays_d = rays_d.reshape(-1, 3)

            termination_prob_list = []
            rendered_weights_list = []
            rendered_depth_uncertainty_list = []
            depth_list = []
            color_list = []

            ray_batch_size = self.ray_batch_size
            gt_depth = gt_depth.reshape(-1)

            for i in range(0, rays_d.shape[0], ray_batch_size):
                rays_d_batch = rays_d[i:i+ray_batch_size]
                rays_o_batch = rays_o[i:i+ray_batch_size]
                if gt_depth is None:
                    ret = self.render_batch_ray(scene_rep, decoders, rays_d_batch, rays_o_batch,
                                                device, truncation, gt_depth=None)
                else:
                    gt_depth_batch = gt_depth[i:i+ray_batch_size]
                    ret = self.render_batch_ray(scene_rep, decoders, rays_d_batch, rays_o_batch,
                                                device, truncation, gt_depth=gt_depth_batch)

                termination_prob, rendered_weights, depth, color, _, _, rendered_depth_uncertainty = ret

                termination_prob_list.append(termination_prob.double())
                rendered_weights_list.append(rendered_weights.double())
                rendered_depth_uncertainty_list.append(rendered_depth_uncertainty.double())
                depth_list.append(depth.double())
                color_list.append(color)

            termination_prob = torch.cat(termination_prob_list, dim=0)
            rendered_weights = torch.cat(rendered_weights_list, dim=0)
            rendered_depth_uncertainty = torch.cat(rendered_depth_uncertainty_list, dim=0)
            depth = torch.cat(depth_list, dim=0)
            color = torch.cat(color_list, dim=0)

            termination_prob = termination_prob.reshape(H, W)
            rendered_weights = rendered_weights.reshape(H, W)
            rendered_depth_uncertainty = rendered_depth_uncertainty.reshape(H, W)
            depth = depth.reshape(H, W)
            color = color.reshape(H, W, 3)

            return depth, color, termination_prob, rendered_weights, rendered_depth_uncertainty