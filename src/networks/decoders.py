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
import torch.nn.functional as F
from src.common import normalize_3d_coordinate
import tinycudann as tcnn

class Decoders(nn.Module):
    """
    Decoders for SDF and RGB.
    Args:
        c_dim: feature dimensions
        hidden_size: hidden size of MLP
        truncation: truncation of SDF
        n_blocks: number of MLP blocks
        learnable_beta: whether to learn beta

    """
    def __init__(self, cfg, c_dim=32, hidden_size=16, truncation=0.08, n_blocks=2, learnable_beta=True):
        super().__init__()

        self.c_dim = c_dim
        self.cfg = cfg
        self.truncation = truncation
        self.n_blocks = n_blocks
        self.tcnn_network = self.cfg['grid']['tcnn_network']

        if cfg['grid_mode'] == 'hash_grid':
            input_channels = c_dim        
        else:
            input_channels = 2 * c_dim   

        if self.tcnn_network:
            self.sdf_decoder = tcnn.Network(
                n_input_dims = input_channels,
                n_output_dims = 1,
                network_config = {
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "Tanh",
                    "n_neurons": hidden_size,
                    "n_hidden_layers": n_blocks-1  
                })

            self.color_decoder = tcnn.Network(
                n_input_dims = input_channels,
                n_output_dims = 3,
                network_config = {
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "Sigmoid",
                    "n_neurons": hidden_size,
                    "n_hidden_layers": n_blocks-1
                })
            
        else:
            ## layers for SDF decoder
            self.linears = nn.ModuleList(
                [nn.Linear(input_channels, hidden_size)] +
                [nn.Linear(hidden_size, hidden_size) for i in range(n_blocks - 1)])

            ## layers for RGB decoder
            self.c_linears = nn.ModuleList(
                [nn.Linear(input_channels, hidden_size)] +
                [nn.Linear(hidden_size, hidden_size)  for i in range(n_blocks - 1)])

            self.output_linear = nn.Linear(hidden_size, 1)
            self.c_output_linear = nn.Linear(hidden_size, 3)

        if learnable_beta:
            self.beta = nn.Parameter(10 * torch.ones(1))
        else:
            self.beta = 10
    
    def sample_hash_grid_feature(self, p_nor, hash_grids_xyz):
        """
        Sample feature from hash_grids
        Args:
            p_nor (tensor): normalized 3D coordinates
            hash_grids_xyz (list): 
           
        Returns:
            feat (tensor): sampled features
        """
        p_nor = torch.clamp(p_nor, min=0, max=1)

        feat = hash_grids_xyz[0](p_nor)   

        return feat

    def get_raw_sdf(self, p_nor, scene_rep):
        """
        Get raw SDF
        Args:
            p_nor (tensor): normalized 3D coordinates
            scene_rep (Tuple): 
        Returns:
            sdf (tensor): raw SDF
        """
        if self.cfg['grid_mode'] == 'hash_grid':
            hash_grids_xyz, c_hash_grids_xyz = scene_rep
            feat = self.sample_hash_grid_feature(p_nor, hash_grids_xyz)   

        h = feat

        if self.tcnn_network:
            sdf = self.sdf_decoder(h).squeeze()
        else: 
            for i, l in enumerate(self.linears):
                h = self.linears[i](h)
                h = F.relu(h, inplace=True)
            sdf = torch.tanh(self.output_linear(h)).squeeze()

        return sdf

    def get_raw_rgb(self, p_nor, scene_rep):
        """
        Get raw RGB
        Args:
            p_nor (tensor): normalized 3D coordinates
            scene_rep (Tuple): all feature grids
        Returns:
            rgb (tensor): raw RGB
        """
        if self.cfg['grid_mode'] == 'hash_grid':
            hash_grids_xyz, c_hash_grids_xyz = scene_rep
            c_feat = self.sample_hash_grid_feature(p_nor, c_hash_grids_xyz)
        
        h = c_feat

        if self.tcnn_network:
            rgb = self.color_decoder(h)
        else:
            for i, l in enumerate(self.c_linears):
                h = self.c_linears[i](h)
                h = F.relu(h, inplace=True)
            rgb = torch.sigmoid(self.c_output_linear(h))

        return rgb
    
    def get_raw_unc(self, p_nor, scene_rep):
        """
        Get raw Uncertainty
        Args:
            p_nor (tensor): normalized 3D coordinates
            scene_rep (Tuple): all feature grids
        Returns:
            rgb (tensor): raw Uncertainty
        """
        if self.cfg['grid_mode'] == 'hash_grid':
            hash_grids_xyz, c_hash_grids_xyz, unc_hash_grids_xyz = scene_rep
            unc_feat = self.sample_hash_grid_feature(p_nor, unc_hash_grids_xyz)

        h = unc_feat

        if self.tcnn_network:
            unc = self.uncertainty_decoder(h)
        else:
            for i, l in enumerate(self.c_linears):
                h = self.c_linears[i](h)
                h = F.relu(h, inplace=True)
            unc = torch.sigmoid(self.c_output_linear(h))

        return unc
    
    def forward(self, p, scene_rep):
        """
        Forward pass
        Args:
            p (tensor): 3D coordinates
            scene_rep (Tuple): all feature grids
        Returns:
            raw (tensor): raw SDF and RGB
        """
        p_shape = p.shape

        if self.cfg['grid_mode'] == 'hash_grid':
            p_nor = p.reshape(-1, 3)       
        else:    
            p_nor = normalize_3d_coordinate(p.clone(), self.bound)

        sdf = self.get_raw_sdf(p_nor, scene_rep)
        rgb = self.get_raw_rgb(p_nor, scene_rep)

        raw = torch.cat([rgb, sdf.unsqueeze(-1)], dim=-1)

        raw = raw.reshape(*p_shape[:-1], -1)

        return raw 
