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

import glob
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from src.common import as_intrinsics_matrix, get_camera_rays
from torch.utils.data import Dataset, Sampler

class SeqSampler(Sampler):
    """
    Sample a sequence of frames from a dataset.

    """
    def __init__(self, n_samples, step, include_last=True):
        self.n_samples = n_samples
        self.step = step
        self.include_last = include_last
    def __iter__(self):
        if self.include_last:
            return iter(list(range(0, self.n_samples, self.step)) + [self.n_samples - 1])
        else:
            return iter(range(0, self.n_samples, self.step))

    def __len__(self) -> int:
        return self.n_samples

def get_dataset(cfg, args, scale, device='cuda:0'):
    return dataset_dict[cfg['dataset']](cfg, args, scale, device=device)

class BaseDataset(Dataset):
    def __init__(self, cfg, args, scale, device='cuda:0'):
        super(BaseDataset, self).__init__()
        self.name = cfg['dataset']
        self.cfg = cfg
        self.device = device
        self.scale = scale
        self.png_depth_scale = cfg['cam']['png_depth_scale']
        self.rays_d = None

        self.H, self.W, self.fx, self.fy, self.cx, self.cy = cfg['cam']['H'], cfg['cam'][
            'W'], cfg['cam']['fx'], cfg['cam']['fy'], cfg['cam']['cx'], cfg['cam']['cy']

        self.distortion = np.array(
            cfg['cam']['distortion']) if 'distortion' in cfg['cam'] else None
        self.crop_size = cfg['cam']['crop_size'] if 'crop_size' in cfg['cam'] else None

        if args.input_folder is None:
            self.input_folder = cfg['data']['input_folder']
        else:
            self.input_folder = args.input_folder

        self.crop_edge = cfg['cam']['crop_edge']

    def __len__(self):
        return self.n_img

    def __getitem__(self, index):
        color_path = self.color_paths[index]
        depth_path = self.depth_paths[index]
        color_data = cv2.imread(color_path)
        depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if self.distortion is not None:
            K = as_intrinsics_matrix([self.fx, self.fy, self.cx, self.cy])
            # undistortion is only applied on color image, not depth!
            color_data = cv2.undistort(color_data, K, self.distortion)

        color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
        color_data = color_data / 255.
        depth_data = depth_data.astype(np.float32) / self.png_depth_scale
        H, W = depth_data.shape
        color_data = cv2.resize(color_data, (W, H))
        color_data = torch.from_numpy(color_data)
        depth_data = torch.from_numpy(depth_data)*self.scale

        if self.crop_size is not None:
            # follow the pre-processing step in lietorch, actually is resize
            color_data = color_data.permute(2, 0, 1)
            color_data = F.interpolate(
                color_data[None], self.crop_size, mode='bilinear', align_corners=True)[0]
            depth_data = F.interpolate(
                depth_data[None, None], self.crop_size, mode='nearest')[0, 0]
            color_data = color_data.permute(1, 2, 0).contiguous()

        edge = self.crop_edge
        if edge > 0:
            # crop image edge, there are invalid value on the edge of the color image
            color_data = color_data[edge:-edge, edge:-edge]
            depth_data = depth_data[edge:-edge, edge:-edge]
        pose = self.poses[index]
        pose[:3, 3] *= self.scale

        if self.crop_size is not None:                # TODO 这一块是我加上的   有可能不同数据集时候有bug 注意
            crop_size = self.cfg['cam']['crop_size']
            sx = crop_size[1] / self.W
            sy = crop_size[0] / self.H
            fx = sx*self.fx
            fy = sy*self.fy
            cx = sx*self.cx
            cy = sy*self.cy
            W = crop_size[1]
            H = crop_size[0]
        else:
            cx = 1*self.cx
            cy = 1*self.cy
            fx = 1*self.fx
            fy = 1*self.fy

        # croping will change H, W, cx, cy, so need to change here
        if edge > 0:
            H -= edge*2
            W -= edge*2
            cx -= edge
            cy -= edge

        if self.rays_d is None:
            self.rays_d = get_camera_rays(H, W, fx, fy, cx, cy)

        # return index, color_data.to(self.device, non_blocking=True), depth_data.to(self.device, non_blocking=True), pose.to(self.device, non_blocking=True)
        return index, color_data, depth_data, pose, self.rays_d

class Replica(BaseDataset):
    def __init__(self, cfg, args, scale, device='cuda:0'
                 ):
        super(Replica, self).__init__(cfg, args, scale, device)
        self.color_paths = sorted(
            glob.glob(f'{self.input_folder}/results/frame*.jpg'))
        self.depth_paths = sorted(
            glob.glob(f'{self.input_folder}/results/depth*.png'))
        self.n_img = len(self.color_paths)
        self.load_poses(f'{self.input_folder}/traj.txt')

    def load_poses(self, path):
        self.poses = []
        with open(path, "r") as f:
            lines = f.readlines()
        for i in range(self.n_img):
            line = lines[i]
            c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
            c2w[:3, 1] *= -1
            c2w[:3, 2] *= -1
            c2w = torch.from_numpy(c2w).float()
            self.poses.append(c2w)

import re           
def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [int(x) if x.isdigit() else x for x in re.split('([0-9]+)', s)]

class RGBDataset(BaseDataset):
    def __init__(self, cfg, args, scale, device='cuda:0'):
        super(RGBDataset, self).__init__(cfg, args, scale, device)
        
        self.color_paths = [os.path.join(self.input_folder, 'images', f) for f in sorted(os.listdir(os.path.join(self.input_folder, 'images')), key=alphanum_key) if f.endswith('png')]
        # if self.cfg["use_generated_depth"] == True:
        #     self.depth_paths = [os.path.join(self.input_folder, 'depth_generated', f) for f in sorted(os.listdir(os.path.join(self.input_folder, 'depth_generated')), key=alphanum_key) if f.endswith('png')]
        #     self.depth_real_paths = [os.path.join(self.input_folder, self.cfg["data"]['depth_folder'], f) for f in sorted(os.listdir(os.path.join(self.input_folder, self.cfg["data"]['depth_folder'])), key=alphanum_key) if f.endswith('png')]
        # else:
        #     self.depth_paths = [os.path.join(self.input_folder, self.cfg["data"]['depth_folder'], f) for f in sorted(os.listdir(os.path.join(self.input_folder, self.cfg["data"]['depth_folder'])), key=alphanum_key) if f.endswith('png')]

        if cfg['data']['depth_folder'] == 'depth':
            self.depth_paths = [os.path.join(self.input_folder, 'depth_gt', f) for f in sorted(os.listdir(os.path.join(self.input_folder, 'depth_gt')), key=alphanum_key) if f.endswith('png')]                       # _gt_noise
        elif cfg['data']['depth_folder'] == 'depth_with_noise': 
            self.depth_paths = [os.path.join(self.input_folder, 'depth_filtered', f) for f in sorted(os.listdir(os.path.join(self.input_folder, 'depth_filtered')), key=alphanum_key) if f.endswith('png')]
        
        self.n_img = len(self.color_paths)
        self.poses, valid_gt_poses = self.load_poses(f'{self.input_folder}/poses.txt')

    def load_poses(self, path):
        file = open(path, "r")
        lines = file.readlines()
        file.close()
        poses = []
        valid = []
        lines_per_matrix = 4
        for i in range(0, len(lines), lines_per_matrix):
            if 'nan' in lines[i]:
                valid.append(False)
                poses.append(np.eye(4, 4, dtype=np.float32).tolist())
            else:
                valid.append(True)
                pose_floats = [[float(x) for x in line.split()] for line in lines[i:i+lines_per_matrix]]
                poses.append(torch.tensor(pose_floats))

        return poses, valid

class ScanNet(BaseDataset):
    def __init__(self, cfg, args, scale, device='cuda:0'
                 ):
        super(ScanNet, self).__init__(cfg, args, scale, device)
        # self.input_folder = os.path.join(self.input_folder, 'frames')
        self.color_paths = sorted(glob.glob(os.path.join(
            self.input_folder, 'color', '*.jpg')), key=lambda x: int(os.path.basename(x)[:-4]))
        self.depth_paths = sorted(glob.glob(os.path.join(
            self.input_folder, 'depth', '*.png')), key=lambda x: int(os.path.basename(x)[:-4]))
        self.load_poses(os.path.join(self.input_folder, 'pose'))
        self.n_img = len(self.color_paths)

    def load_poses(self, path):
        self.poses = []
        pose_paths = sorted(glob.glob(os.path.join(path, '*.txt')),
                            key=lambda x: int(os.path.basename(x)[:-4]))
        for pose_path in pose_paths:
            with open(pose_path, "r") as f:
                lines = f.readlines()
            ls = []
            for line in lines:
                l = list(map(float, line.split(' ')))
                ls.append(l)
            c2w = np.array(ls).reshape(4, 4)
            c2w[:3, 1] *= -1
            c2w[:3, 2] *= -1
            c2w = torch.from_numpy(c2w).float()
            self.poses.append(c2w)

class TUM_RGBD(BaseDataset):
    def __init__(self, cfg, args, scale, device='cuda:0'
                 ):
        super(TUM_RGBD, self).__init__(cfg, args, scale, device)
        self.color_paths, self.depth_paths, self.poses = self.loadtum(
            self.input_folder, frame_rate=32)
        self.n_img = len(self.color_paths)

    def parse_list(self, filepath, skiprows=0):
        """ read list data """
        data = np.loadtxt(filepath, delimiter=' ',
                          dtype=np.unicode_, skiprows=skiprows)
        return data

    def associate_frames(self, tstamp_image, tstamp_depth, tstamp_pose, max_dt=0.08):
        """ pair images, depths, and poses """
        associations = []
        for i, t in enumerate(tstamp_image):
            if tstamp_pose is None:
                j = np.argmin(np.abs(tstamp_depth - t))
                if (np.abs(tstamp_depth[j] - t) < max_dt):
                    associations.append((i, j))

            else:
                j = np.argmin(np.abs(tstamp_depth - t))
                k = np.argmin(np.abs(tstamp_pose - t))

                if (np.abs(tstamp_depth[j] - t) < max_dt) and \
                        (np.abs(tstamp_pose[k] - t) < max_dt):
                    associations.append((i, j, k))

        return associations

    def loadtum(self, datapath, frame_rate=-1):
        """ read video data in tum-rgbd format """
        if os.path.isfile(os.path.join(datapath, 'groundtruth.txt')):
            pose_list = os.path.join(datapath, 'groundtruth.txt')
        elif os.path.isfile(os.path.join(datapath, 'pose.txt')):
            pose_list = os.path.join(datapath, 'pose.txt')

        image_list = os.path.join(datapath, 'rgb.txt')
        depth_list = os.path.join(datapath, 'depth.txt')

        image_data = self.parse_list(image_list)
        depth_data = self.parse_list(depth_list)
        pose_data = self.parse_list(pose_list, skiprows=1)
        pose_vecs = pose_data[:, 1:].astype(np.float64)

        tstamp_image = image_data[:, 0].astype(np.float64)
        tstamp_depth = depth_data[:, 0].astype(np.float64)
        tstamp_pose = pose_data[:, 0].astype(np.float64)
        associations = self.associate_frames(
            tstamp_image, tstamp_depth, tstamp_pose)

        indicies = [0]
        for i in range(1, len(associations)):
            t0 = tstamp_image[associations[indicies[-1]][0]]
            t1 = tstamp_image[associations[i][0]]
            if t1 - t0 > 1.0 / frame_rate:
                indicies += [i]

        images, poses, depths, intrinsics = [], [], [], []
        inv_pose = None
        for ix in indicies:
            (i, j, k) = associations[ix]
            images += [os.path.join(datapath, image_data[i, 1])]
            depths += [os.path.join(datapath, depth_data[j, 1])]
            c2w = self.pose_matrix_from_quaternion(pose_vecs[k])
            if inv_pose is None:
                inv_pose = np.linalg.inv(c2w)
                c2w = np.eye(4)
            else:
                c2w = inv_pose@c2w
            c2w[:3, 1] *= -1
            c2w[:3, 2] *= -1
            c2w = torch.from_numpy(c2w).float()
            poses += [c2w]

        return images, depths, poses

    def pose_matrix_from_quaternion(self, pvec):
        """ convert 4x4 pose matrix to (t, q) """
        from scipy.spatial.transform import Rotation

        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat(pvec[3:]).as_matrix()
        pose[:3, 3] = pvec[:3]
        return pose
    
class Azure(BaseDataset):
    def __init__(self, cfg, args, scale, device='cuda:0'
                 ):
        super(Azure, self).__init__(cfg, args, scale, device)
        self.color_paths = sorted(
            glob.glob(os.path.join(self.input_folder, 'color', '*.jpg')))
        self.depth_paths = sorted(
            glob.glob(os.path.join(self.input_folder, 'depth', '*.png')))
        self.n_img = len(self.color_paths)
        self.load_poses(os.path.join(
            self.input_folder, 'scene', 'trajectory.log'))

    def load_poses(self, path):
        self.poses = []
        if os.path.exists(path):
            with open(path) as f:
                content = f.readlines()

                # Load .log file.
                for i in range(0, len(content), 5):
                    # format %d (src) %d (tgt) %f (fitness)
                    data = list(map(float, content[i].strip().split(' ')))
                    ids = (int(data[0]), int(data[1]))
                    fitness = data[2]

                    # format %f x 16
                    c2w = np.array(
                        list(map(float, (''.join(
                            content[i + 1:i + 5])).strip().split()))).reshape((4, 4))

                    c2w[:3, 1] *= -1
                    c2w[:3, 2] *= -1
                    c2w = torch.from_numpy(c2w).float()
                    self.poses.append(c2w)
        else:
            for i in range(self.n_img):
                c2w = np.eye(4)
                c2w = torch.from_numpy(c2w).float()
                self.poses.append(c2w)


dataset_dict = {
    "replica": Replica,
    "scannet": ScanNet,
    "tumrgbd": TUM_RGBD,
    "azure": Azure,
    "systheticrgbd": RGBDataset,
}