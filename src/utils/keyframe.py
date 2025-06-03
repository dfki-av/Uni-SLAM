import torch
import numpy as np
import random
import os
import json
from src.common import get_samples
                        
class KeyFrameDatabase(object):
    def __init__(self, cfg, n_img, num_rays_to_save=None) -> None:

        self.cfg = cfg
        self.device = cfg['device']
            
        self.H = cfg['cam']['H']
        self.W = cfg['cam']['W']
        self.keyframe_every = cfg['mapping']['keyframe_every']
        self.keyframe_selection_method = self.cfg['mapping']['keyframe_selection_method']
        self.num_rays_to_save = num_rays_to_save
        self.output = cfg['data']['output']

        total_num_kf = int(n_img // self.cfg['mapping']['keyframe_every'] + 1 +1)  
        print('#kf:', total_num_kf)
        
        self.keyframe_list = []
        if self.keyframe_selection_method == 'global':
            self.keyframe_dict = {'idx': torch.zeros((total_num_kf)),
                                  'gt_c2w': torch.zeros((total_num_kf, 4, 4)), 
                                  'est_c2w': torch.zeros((total_num_kf, 4, 4)).to(self.device), 
                                  'color': torch.zeros((total_num_kf, self.H, self.W, 3)).to(self.device), 
                                  'gt_depth': torch.zeros((total_num_kf, self.H, self.W)).to(self.device),
                                  'keyframe_list': self.keyframe_list}

    def __len__(self):
        return len(self.keyframe_list)
    

    def add_keyframe(self, idx, cur_c2w, gt_c2w, gt_depth, gt_color):

        self.keyframe_list.append(idx)

        if self.keyframe_selection_method == 'global':
            i = idx // self.cfg['mapping']['keyframe_every']
            self.keyframe_dict['idx'][i] = idx
            self.keyframe_dict['gt_c2w'][i] = gt_c2w 
            self.keyframe_dict['est_c2w'][i] = cur_c2w.clone()
            self.keyframe_dict['color'][i] = gt_color.to(self.device)
            self.keyframe_dict['gt_depth'][i] = gt_depth.to(self.device) 

    def get_optimize_frame(self, idx, optimize_frame, cur_gt_depth, cur_gt_color, cur_c2w, gt_cur_c2w):

        if self.cfg['mapping']['keyframe_selection_method'] == 'global':
            
            num_kf = self.__len__()   

            gt_depths = self.keyframe_dict['gt_depth'][:num_kf+1]
            gt_colors = self.keyframe_dict['color'][:num_kf+1]
            c2ws = self.keyframe_dict['est_c2w'][:num_kf+1].clone()
            gt_c2ws = self.keyframe_dict['gt_c2w'][:num_kf+1]

            gt_depths[num_kf] = cur_gt_depth
            gt_colors[num_kf] = cur_gt_color 
            c2ws[num_kf] = cur_c2w.clone()
            gt_c2ws[num_kf] = gt_cur_c2w
            

        return gt_depths, gt_colors, c2ws, gt_c2ws 
    

    def keyframe_selection_all(self, len_keyframe_dict):
        """
        这里返回的是除了最后两个frame外的所有idx

        Return: list [0, 1, 2, 3 ...]
        """

        return list(range(0, len_keyframe_dict))