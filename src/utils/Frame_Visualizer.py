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

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from src.common import cam_pose_to_matrix
from skimage.metrics import mean_squared_error

class Frame_Visualizer(object):
    """
    Visualizes itermediate results, render out depth and color images.
    It can be called per iteration, which is good for debuging (to see how each tracking/mapping iteration performs).
    Args:
        freq (int): frequency of visualization.
        inside_freq (int): frequency of visualization inside each iteration.
        vis_dir (str): directory to save the visualization results.
        renderer (Renderer): renderer.
        truncation (float): truncation distance.
        verbose (bool): whether to print out the visualization results.
        device (str): device.
    """

    def __init__(self, freq, inside_freq, vis_dir, renderer, truncation, verbose, device='cuda:0'):
        self.freq = freq
        self.device = device
        self.vis_dir = vis_dir
        self.verbose = verbose
        self.renderer = renderer
        self.inside_freq = inside_freq
        self.truncation = truncation
        os.makedirs(f'{vis_dir}', exist_ok=True)

    def mse2psnr(self, MSE, MAX=None):
        '''
        MSE to PSNR(Peak Signal-to-Noise Ratio)   
        '''
        MSE = torch.tensor(MSE)
        if MAX:
            MAX = torch.tensor(MAX)
            psnr = 10 * torch.log10(MAX**2 / MSE)                                
        # psnr = -10. * torch.log(x) / torch.log(torch.Tensor([10.])).to(x)  #  coslam
        else:
            psnr = -10. * torch.log10(MSE)
        psnr = psnr.numpy()

        return psnr

    def save_mapping_imgs(self, output_dir, idx, iter, gt_depth, gt_color, c2w_or_camera_tensor, scene_rep, decoders):
        vis_dir=os.path.join(output_dir, 'render_img_{}'.format(idx))
        os.makedirs(f'{vis_dir}', exist_ok=True)

        with torch.no_grad():
            
            gt_depth_np = gt_depth.squeeze(0).cpu().numpy()
            gt_color_np = gt_color.squeeze(0).cpu().numpy()

            if c2w_or_camera_tensor.shape[-1] > 4: ## 6od
                c2w = cam_pose_to_matrix(c2w_or_camera_tensor.clone().detach()).squeeze()
            else:
                c2w = c2w_or_camera_tensor.squeeze().detach()

            depth, color, termination_prob, rendered_weights, rendered_depth_uncertainty = self.renderer.render_img(scene_rep, decoders, c2w, self.truncation,
                                                    self.device, gt_depth=gt_depth)

            rendered_weights_np = rendered_weights.detach().cpu().numpy()
            rendered_depth_uncertainty_np = rendered_depth_uncertainty.detach().cpu().numpy()
            depth_np = depth.detach().cpu().numpy()
            color_np = color.detach().cpu().numpy()

            color_mse = mean_squared_error(gt_color_np, color_np)    
            color_psnr_value = self.mse2psnr(color_mse)
            color_titles = 'Color mse: {:.4f}  PSNR: {:.4f}'.format(color_mse, color_psnr_value)

            gt_color_np = np.clip(gt_color_np, 0, 1)
            color_np = np.clip(color_np, 0, 1)

            image_path = os.path.join(vis_dir, f'{iter}.png')
            plt.imsave(image_path, color_np)

            psnr_record_path = os.path.join(vis_dir, 'psnr_record.txt')
            with open(psnr_record_path, 'a') as f:
                f.write(color_titles + '\n')



    def save_imgs(self, idx, iter, gt_depth, gt_color, c2w_or_camera_tensor, scene_rep, decoders):
        """
        Visualization of depth and color images and save to file.
        Args:
            idx (int): current frame index.
            iter (int): the iteration number.
            gt_depth (tensor): ground truth depth image of the current frame.
            gt_color (tensor): ground truth color image of the current frame.
            c2w_or_camera_tensor (tensor): camera pose, represented in 
                camera to world matrix or quaternion and translation tensor.
            scene_rep (Tuple): feature grids.
            decoders (torch.nn.Module): decoders for TSDF and color.
        """
        with torch.no_grad():
            if (idx % self.freq == 0):
                gt_depth_np = gt_depth.squeeze(0).cpu().numpy()
                gt_color_np = gt_color.squeeze(0).cpu().numpy()

                if c2w_or_camera_tensor.shape[-1] > 4: ## 6od
                    c2w = cam_pose_to_matrix(c2w_or_camera_tensor.clone().detach()).squeeze()
                else:
                    c2w = c2w_or_camera_tensor.squeeze().detach()

                depth, color, termination_prob, rendered_weights, rendered_depth_uncertainty = self.renderer.render_img(scene_rep, decoders, c2w, self.truncation,
                                                        self.device, gt_depth=gt_depth)
                
                termination_prob_np = termination_prob.detach().cpu().numpy()
                rendered_weights_np = rendered_weights.detach().cpu().numpy()
                rendered_depth_uncertainty_np = rendered_depth_uncertainty.detach().cpu().numpy()
                depth_np = depth.detach().cpu().numpy()
                color_np = color.detach().cpu().numpy()

                self.save_wo_uncertainty(idx, gt_depth_np, depth_np, gt_color_np, color_np, termination_prob_np, rendered_weights_np, rendered_depth_uncertainty_np)


    def save_wo_uncertainty(self, idx, gt_depth_np, depth_np, gt_color_np, color_np, 
                            termination_prob_np, rendered_weights_np=None, rendered_depth_uncertainty_np=None, gt_depth_real=None):
                                
        """
        
        """
        if gt_depth_real!=None:
            gt_depth_real_np = gt_depth_real.squeeze(0).cpu().numpy()  

        if gt_depth_real!=None:
            depth_residual = np.abs(gt_depth_real_np - depth_np)
            depth_residual[gt_depth_real_np == 0.0] = 0.0    
        else:
            depth_residual = np.abs(gt_depth_np - depth_np)
            depth_residual[gt_depth_np == 0.0] = 0.0
        color_residual = np.abs(gt_color_np - color_np)      


        #######   Code for the Color Scale Bar   #######
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable

        # Define the colormap
        cmap = plt.get_cmap('viridis')
        # Define the range for your colorbar
        norm = Normalize(vmin=0, vmax=0.01)  # norm = Normalize(vmin=0, vmax=0.25) 
        # Create a figure for the colorbar
        fig, ax = plt.subplots(figsize=(0.35, 4), dpi=300)
        # Create the colorbar
        cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=ax)
        cb.set_label('Uncertainty Value')           # cb.set_label('ATE Value')

        # Set the label sizes
        cb.ax.yaxis.label.set_size(10)
        cb.ax.yaxis.set_tick_params(labelsize=8)
        cb.ax.yaxis.set_tick_params(right=False)  # This line hides the right-side tick marks

        # Save the colorbar figure
        fig.savefig(os.path.join(self.vis_dir, 'uncertainty_bar.png'), bbox_inches='tight')
        plt.close(fig)

        h, w = depth_np.shape
        fig, axs = plt.subplots(2, 4)
        fig.tight_layout()

        if gt_depth_real!=None:
            max_depth = np.max(gt_depth_real_np)
            depth_mask = (gt_depth_real_np > 0)
            depth_mse = mean_squared_error(gt_depth_real_np[depth_mask], depth_np[depth_mask])  
        else:
            max_depth = np.max(gt_depth_np)
            depth_mask = (gt_depth_np > 0)
            depth_mse = mean_squared_error(gt_depth_np[depth_mask], depth_np[depth_mask])

        depth_psnr_value = self.mse2psnr(depth_mse)
        depth_titles = 'Depth mse: {:.4f}  PSNR: {:.4f}'.format(depth_mse, depth_psnr_value)
        fig.text(0.5, 0.95, depth_titles, ha='center', fontsize=8)    
    
        axs[0, 0].imshow(gt_depth_np, cmap="plasma", vmin=0, vmax=max_depth)
        axs[0, 0].set_title('Input Depth')
        axs[0, 0].set_xticks([])
        axs[0, 0].set_yticks([])
        axs[0, 1].imshow(depth_np, cmap="plasma", vmin=0, vmax=max_depth)
        axs[0, 1].set_title('Generated Depth')
        axs[0, 1].set_xticks([])
        axs[0, 1].set_yticks([])
        # axs[0, 2].imshow(depth_residual, cmap="plasma", vmin=0, vmax=max_depth)
        # axs[0, 2].set_title('Depth Residual')
        # axs[0, 2].imshow(termination_prob_np, cmap="plasma", vmin=0, vmax=1)
        axs[0, 2].imshow(termination_prob_np, cmap="viridis", vmin=0, vmax=1)
        axs[0, 2].set_title('Termination Prob')
        axs[0, 2].set_xticks([])
        axs[0, 2].set_yticks([])
        # axs[0, 3].imshow(np.zeros((h, w)))
        axs[0, 3].imshow(rendered_weights_np, cmap="viridis", vmin=0, vmax=0.01)   
        # axs[0, 3].imshow(rendered_weights_np, cmap="viridis", vmin=0, vmax=1)   
        # axs[0, 3].imshow(rendered_weights_np, cmap="jet", vmin=0, vmax=0.0001)
        # axs[0, 3].imshow(rendered_weights_np, cmap="plasma", vmin=0, vmax=0.00001)
        axs[0, 3].set_title('Rendered Weights')
        axs[0, 3].set_xticks([])
        axs[0, 3].set_yticks([])

        color_mse = mean_squared_error(gt_color_np, color_np)   
        color_psnr_value = self.mse2psnr(color_mse)
        color_titles = 'Color mse: {:.4f}  PSNR: {:.4f}'.format(color_mse, color_psnr_value)
        fig.text(0.5, 0.52, color_titles, ha='center', fontsize=8)
    
        gt_color_np = np.clip(gt_color_np, 0, 1)
        color_np = np.clip(color_np, 0, 1)
        color_residual = np.clip(color_residual, 0, 1)
        axs[1, 0].imshow(gt_color_np, cmap="plasma")
        axs[1, 0].set_title('Input RGB')
        axs[1, 0].set_xticks([])
        axs[1, 0].set_yticks([])
        axs[1, 1].imshow(color_np, cmap="plasma")
        axs[1, 1].set_title('Generated RGB')
        axs[1, 1].set_xticks([])
        axs[1, 1].set_yticks([])
        axs[1, 2].imshow(color_residual, cmap="plasma")
        axs[1, 2].set_title('RGB Residual')
        axs[1, 2].set_xticks([])
        axs[1, 2].set_yticks([])

        # axs[1, 3].imshow(np.zeros((h, w)))
        axs[1, 3].imshow(rendered_depth_uncertainty_np, cmap="jet", vmin=0, vmax=1.0)
        axs[1, 3].set_title('Depth Uncertainty')
        axs[1, 3].set_xticks([])
        axs[1, 3].set_yticks([])
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.savefig(f'{self.vis_dir}/{idx:05d}.jpg', bbox_inches='tight', pad_inches=0.2, dpi=300)
        plt.cla()
        plt.clf()
        plt.close(fig)   

        if self.verbose:
            print(f'Saved rendering visualization of color/depth image at {self.vis_dir}/{idx:05d}.jpg')
