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

class Logger(object):
    """
    Save checkpoints to file.

    """

    def __init__(self, unislam):
        self.verbose = unislam.verbose
        self.ckptsdir = unislam.ckptsdir
        self.gt_c2w_list = unislam.gt_c2w_list
        self.shared_decoders = unislam.shared_decoders
        self.estimate_c2w_list = unislam.estimate_c2w_list
        self.tracking_rendered_weight_list = unislam.tracking_rendered_weight_list
        self.addtional_map_records = unislam.addtional_map_records

    def log(self, idx, keyframe_list):
        path = os.path.join(self.ckptsdir, '{:05d}.tar'.format(idx))
        torch.save({
            'decoder_state_dict': self.shared_decoders.state_dict(),
            'gt_c2w_list': self.gt_c2w_list,
            'estimate_c2w_list': self.estimate_c2w_list,
            'keyframe_list': keyframe_list,
            'idx': idx,
            'tracking_rendered_weight_list': self.tracking_rendered_weight_list,
            'addtional_map_records': self.addtional_map_records
        }, path, _use_new_zipfile_serialization=False)

        if self.verbose:
            print('Saved checkpoints at', path)
