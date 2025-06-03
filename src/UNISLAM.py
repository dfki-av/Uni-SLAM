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
import time

import numpy as np
import torch
import torch.multiprocessing
import torch.multiprocessing as mp
import tinycudann as tcnn

from src import config
from src.Mapper import Mapper
from src.Tracker import Tracker
from src.utils.datasets import get_dataset
from src.utils.Logger import Logger
from src.utils.Mesher import Mesher
from src.utils.Renderer import Renderer

torch.multiprocessing.set_sharing_strategy('file_system')

class UNISLAM():
    """
    ESLAM main class.
    Mainly allocate shared resources, and dispatch mapping and tracking processes.
    Args:
        cfg (dict): config dict
        args (argparse.Namespace): arguments
    """

    def __init__(self, cfg, args):

        self.cfg = cfg
        self.args = args

        self.verbose = cfg['verbose']
        self.device = cfg['device']
        self.dataset = cfg['dataset']
        self.truncation = cfg['model']['truncation']

        if args.output is None:
            self.output = cfg['data']['output']
        else:
            self.output = args.output
        import os
        self.ckptsdir = os.path.join(self.output, 'ckpts')
        os.makedirs(self.output, exist_ok=True)
        os.makedirs(self.ckptsdir, exist_ok=True)
        os.makedirs(f'{self.output}/mesh', exist_ok=True)
        self.H, self.W, self.fx, self.fy, self.cx, self.cy = cfg['cam']['H'], cfg['cam'][
            'W'], cfg['cam']['fx'], cfg['cam']['fy'], cfg['cam']['cx'], cfg['cam']['cy']
        self.update_cam()

        model = config.get_model(cfg)
        self.shared_decoders = model

        self.scale = cfg['scale']

        self.load_bound(cfg)

        if cfg['grid_mode'] == 'hash_grid':
            hash_grids_xyz = []
            c_hash_grids_xyz = []
            unc_hash_grids_xyz = []
            # Sparse parametric encoding (SDF)
            self.embed_fn, self.input_ch = self.get_encoder(encoding_method=cfg['grid']['enc'], log2_hashmap_size=cfg['grid']['hash_size_sdf'], desired_resolution=self.resolution_sdf)
            hash_grids_xyz.append(self.embed_fn)
            self.shared_hash_grids_xyz = hash_grids_xyz

            # Sparse parametric encoding (Color)
            self.embed_fn_color, self.input_ch_color = self.get_encoder(encoding_method=cfg['grid']['enc'], log2_hashmap_size=cfg['grid']['hash_size_color'], desired_resolution=self.resolution_color)
            c_hash_grids_xyz.append(self.embed_fn_color)
            self.shared_c_hash_grids_xyz = c_hash_grids_xyz

        # need to use spawn
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

        self.frame_reader = get_dataset(cfg, args, self.scale)
        self.n_img = len(self.frame_reader)
        self.estimate_c2w_list = torch.zeros((self.n_img, 4, 4), device=self.device)
        self.estimate_c2w_list.share_memory_()
        self.tracking_rendered_weight_list = torch.zeros((self.n_img)).share_memory_()
        self.addtional_map_records = torch.zeros((self.n_img), device=self.device)
        self.addtional_map_records.share_memory_()

        self.gt_c2w_list = torch.zeros((self.n_img, 4, 4))
        self.gt_c2w_list.share_memory_()
        self.idx = torch.zeros((1)).int()
        self.idx.share_memory_()
        self.mapping_first_frame = torch.zeros((1)).int()
        self.mapping_first_frame.share_memory_()

        # the id of the newest frame Mapper is processing
        self.mapping_idx = torch.zeros((1)).int()
        self.mapping_idx.share_memory_()
        self.mapping_cnt = torch.zeros((1)).int()  # counter for mapping
        self.mapping_cnt.share_memory_()
        self.LC_cnt = torch.zeros((1)).int()  # counter for mapping
        self.LC_cnt.share_memory_()

        self.tracking_back = torch.tensor([0])  
        self.tracking_back.share_memory_()
        self.m_iters = torch.tensor([cfg['mapping']['iters']])  
        self.m_iters.share_memory_()
        self.t_iters = torch.tensor(self.cfg['tracking']['iters'])  
        self.t_iters.share_memory_()

        ## Moving feature grids and decoders to the processing device
        if cfg['grid_mode'] == 'hash_grid':
            for i, grid in enumerate(self.shared_hash_grids_xyz):
                    grid = grid.to(self.device).share_memory()
                    self.shared_hash_grids_xyz[i] = grid
            
            for i, grid in enumerate(self.shared_c_hash_grids_xyz):
                    grid = grid.to(self.device).share_memory()
                    self.shared_c_hash_grids_xyz[i] = grid

        self.shared_decoders = self.shared_decoders.to(self.device)
        self.shared_decoders.share_memory()

        self.renderer = Renderer(cfg, self)
        self.mesher = Mesher(cfg, args, self)
        self.logger = Logger(self)
        self.mapper = Mapper(cfg, args, self)
        self.tracker = Tracker(cfg, args, self)
        self.print_output_desc()

        import yaml
        output_path = os.path.join(cfg['data']['output'], "config.yaml")
        with open(output_path, 'w') as f:
            yaml.dump(cfg, f)

        import shutil
        import os
        src_folder = 'src'
        dst_folder = os.path.join(cfg['data']['output'], 'src')
       
        if os.path.exists(dst_folder):
            shutil.rmtree(dst_folder)
        shutil.copytree(src_folder, dst_folder)

    def print_output_desc(self):
        print(f"INFO: The output folder is {self.output}")
        print(
            f"INFO: The GT, generated and residual depth/color images can be found under " +
            f"{self.output}/tracking_vis/ and {self.output}/mapping_vis/")
        print(f"INFO: The mesh can be found under {self.output}/mesh/")
        print(f"INFO: The checkpoint can be found under {self.output}/ckpt/")

    def update_cam(self):
        """
        Update the camera intrinsics according to pre-processing config, 
        such as resize or edge crop.
        """
        # resize the input images to crop_size (variable name used in lietorch)
        if 'crop_size' in self.cfg['cam']:
            crop_size = self.cfg['cam']['crop_size']
            sx = crop_size[1] / self.W
            sy = crop_size[0] / self.H
            self.fx = sx*self.fx
            self.fy = sy*self.fy
            self.cx = sx*self.cx
            self.cy = sy*self.cy
            self.W = crop_size[1]
            self.H = crop_size[0]

        # croping will change H, W, cx, cy, so need to change here
        if self.cfg['cam']['crop_edge'] > 0:
            self.H -= self.cfg['cam']['crop_edge']*2
            self.W -= self.cfg['cam']['crop_edge']*2
            self.cx -= self.cfg['cam']['crop_edge']
            self.cy -= self.cfg['cam']['crop_edge']
    
    def get_resolution(self, cfg):
        '''
        Get the resolution of the grid
        '''
        dim_max = (self.bound[:,1] - self.bound[:,0]).max()
   
        self.resolution_sdf = int(dim_max / cfg['grid']['voxel_sdf'])     
        self.resolution_color = int(dim_max / cfg['grid']['voxel_color'])
        
        
        print('SDF resolution:', self.resolution_sdf)
        print('Color resolution:', self.resolution_color)

    def load_bound(self, cfg):
        """
        Pass the scene bound parameters to different decoders and self.

        Args:
            cfg (dict): parsed config dict.
        """

        # scale the bound if there is a global scaling factor
        self.bound = torch.from_numpy(np.array(cfg['mapping']['bound'])*self.scale).float()
        bound_dividable = cfg['planes_res']['bound_dividable']
        # enlarge the bound a bit to allow it dividable by bound_dividable
        self.bound[:, 1] = (((self.bound[:, 1]-self.bound[:, 0]) /
                            bound_dividable).int()+1)*bound_dividable+self.bound[:, 0]
        self.shared_decoders.bound = self.bound

        if cfg['grid_mode']=='hash_grid':
            self.get_resolution(cfg)

    def get_encoder(self, encoding_method, input_dim=3,
                    degree=4, n_bins=16, n_frequencies=12,
                    n_levels=16, level_dim=2, 
                    base_resolution=16, log2_hashmap_size=19, 
                    desired_resolution=512):
        """
        Args:
            n_levels: number of levels
            log2_hashmap_size: hash table size
            base_resolution: coarse resolution
            desired_resolution: finest resolution
            level_dim: Number of feature dimensions per entry
        """
        
        # Sparse grid encoding
        if 'hash' in encoding_method.lower() or 'tiled' in encoding_method.lower():
            print('Hash size', log2_hashmap_size)
            per_level_scale = np.exp2(np.log2(desired_resolution  / n_levels) / (n_levels - 1))    # function (3)
            embed = tcnn.Encoding(
                n_input_dims=input_dim,
                encoding_config={
                    "otype": 'HashGrid',
                    "n_levels": n_levels,                        
                    "n_features_per_level": level_dim,           
                    "log2_hashmap_size": log2_hashmap_size,      
                    "base_resolution": base_resolution,         
                    "per_level_scale": per_level_scale
                },
                dtype=torch.float
            )
            out_dim = embed.n_output_dims                       

        elif encoding_method == 'freq':
            pass

        return embed, out_dim

    def tracking(self, rank):
        """
        Tracking Thread.

        Args:
            rank (int): Thread ID.
        """

        # should wait until the mapping of first frame is finished
        while True:
            if self.mapping_first_frame[0] == 1:
                break
            time.sleep(1)

        self.tracker.run()

    def mapping(self, rank):
        """
        Mapping Thread.

        Args:
            rank (int): Thread ID.
        """

        self.mapper.run()

    def run(self):
        """
        Dispatch Threads.
        """

        processes = []
        for rank in range(0, 2):
            if rank == 0:
                p = mp.Process(target=self.tracking, args=(rank, ))
            elif rank == 1:
                p = mp.Process(target=self.mapping, args=(rank, ))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

# This part is required by torch.multiprocessing
if __name__ == '__main__':
    pass
