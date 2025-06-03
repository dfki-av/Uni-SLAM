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
import random
import os
import json

import numpy as np
import open3d as o3d
import torch
import trimesh
from scipy.spatial import cKDTree as KDTree

def normalize(x):
    return x / np.linalg.norm(x)

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

def completion_ratio(gt_points, rec_points, dist_th=0.01):
    gen_points_kd_tree = KDTree(rec_points)
    distances, _ = gen_points_kd_tree.query(gt_points)
    comp_ratio = np.mean((distances < dist_th).astype('float'))
    return comp_ratio


def accuracy(gt_points, rec_points):
    gt_points_kd_tree = KDTree(gt_points)
    distances, _ = gt_points_kd_tree.query(rec_points)
    acc = np.mean(distances)
    return acc


def completion(gt_points, rec_points):
    gt_points_kd_tree = KDTree(rec_points)
    distances, _ = gt_points_kd_tree.query(gt_points)
    comp = np.mean(distances)
    return comp


def get_align_transformation(rec_meshfile, gt_meshfile):
    """
    Get the transformation matrix to align the reconstructed mesh to the ground truth mesh.
    """
    o3d_rec_mesh = o3d.io.read_triangle_mesh(rec_meshfile)
    o3d_gt_mesh = o3d.io.read_triangle_mesh(gt_meshfile)
    o3d_rec_pc = o3d.geometry.PointCloud(points=o3d_rec_mesh.vertices)
    o3d_gt_pc = o3d.geometry.PointCloud(points=o3d_gt_mesh.vertices)
    trans_init = np.eye(4)
    threshold = 0.1
    reg_p2p = o3d.pipelines.registration.registration_icp(
        o3d_rec_pc, o3d_gt_pc, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    transformation = reg_p2p.transformation
    return transformation


def check_proj(points, W, H, fx, fy, cx, cy, c2w):
    """
    Check if points can be projected into the camera view.

    """
    c2w = c2w.copy()
    c2w[:3, 1] *= -1.0
    c2w[:3, 2] *= -1.0
    points = torch.from_numpy(points).cuda().clone()
    w2c = np.linalg.inv(c2w)
    w2c = torch.from_numpy(w2c).cuda().float()
    K = torch.from_numpy(
        np.array([[fx, .0, cx], [.0, fy, cy], [.0, .0, 1.0]]).reshape(3, 3)).cuda()
    ones = torch.ones_like(points[:, 0]).reshape(-1, 1).cuda()
    homo_points = torch.cat(
        [points, ones], dim=1).reshape(-1, 4, 1).cuda().float()  # (N, 4)
    cam_cord_homo = w2c@homo_points  # (N, 4, 1)=(4,4)*(N, 4, 1)
    cam_cord = cam_cord_homo[:, :3]  # (N, 3, 1)
    cam_cord[:, 0] *= -1
    uv = K.float()@cam_cord.float()
    z = uv[:, -1:]+1e-5
    uv = uv[:, :2]/z
    uv = uv.float().squeeze(-1).cpu().numpy()
    edge = 0
    mask = (0 <= -z[:, 0, 0].cpu().numpy()) & (uv[:, 0] < W -
                                               edge) & (uv[:, 0] > edge) & (uv[:, 1] < H-edge) & (uv[:, 1] > edge)
    return mask.sum() > 0


def calc_3d_metric(rec_meshfile, gt_meshfile, align=True, num_points=450000, dist_th=0.01):
    """
    3D reconstruction metric.

    """
    mesh_rec = trimesh.load(rec_meshfile, process=False)
    mesh_gt = trimesh.load(gt_meshfile, process=False)

    if align:
        transformation = get_align_transformation(rec_meshfile, gt_meshfile)
        mesh_rec = mesh_rec.apply_transform(transformation)

    rec_pc = trimesh.sample.sample_surface(mesh_rec, num_points)
    rec_pc_tri = trimesh.PointCloud(vertices=rec_pc[0])
    gt_pc = trimesh.sample.sample_surface(mesh_gt, num_points)
    gt_pc_tri = trimesh.PointCloud(vertices=gt_pc[0])
    accuracy_rec = accuracy(gt_pc_tri.vertices, rec_pc_tri.vertices)
    completion_rec = completion(gt_pc_tri.vertices, rec_pc_tri.vertices)
    completion_ratio_rec = completion_ratio(
        gt_pc_tri.vertices, rec_pc_tri.vertices, dist_th=dist_th)
    accuracy_rec *= 100  # convert to cm
    completion_rec *= 100  
    completion_ratio_rec *= 100  
    print('accuracy: ', accuracy_rec)
    print('completion: ', completion_rec)
    print('completion ratio < {}: {}'.format(dist_th, completion_ratio_rec))

    return {
        'accuracy': round(accuracy_rec, 2), 
        'completion': round(completion_rec, 2),
        'completion ratio': round(completion_ratio_rec, 2)}


def get_cam_position(gt_meshfile):
    mesh_gt = trimesh.load(gt_meshfile)
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh_gt)
    extents[2] *= 0.7
    extents[1] *= 0.7
    extents[0] *= 0.3
    transform = np.linalg.inv(to_origin)
    transform[2, 3] += 0.4
    return extents, transform


def calc_2d_metric(rec_meshfile, gt_meshfile, align=True, n_imgs=1000):
    """
    2D reconstruction metric, depth L1 loss.

    """
    H = 500
    W = 500
    focal = 300
    fx = focal
    fy = focal
    cx = H/2.0-0.5
    cy = W/2.0-0.5

    gt_mesh = o3d.io.read_triangle_mesh(gt_meshfile)
    rec_mesh = o3d.io.read_triangle_mesh(rec_meshfile)
    # unseen_gt_pointcloud_file = gt_meshfile.replace('.ply', '_pc_unseen.npy')
    unseen_gt_pointcloud_file = gt_meshfile.replace('_culled.ply', '_pc_unseen.npy')
    pc_unseen = np.load(unseen_gt_pointcloud_file)
    if align:
        transformation = get_align_transformation(rec_meshfile, gt_meshfile)
        rec_mesh = rec_mesh.transform(transformation)

    # get vacant area inside the room
    extents, transform = get_cam_position(gt_meshfile)

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=W, height=H)
    vis.get_render_option().mesh_show_back_face = True
    errors = []
    for i in range(n_imgs):
        while True:
            # sample view, and check if unseen region is not inside the camera view
            # if inside, then needs to resample
            up = [0, 0, -1]
            origin = trimesh.sample.volume_rectangular(
                extents, 1, transform=transform)
            origin = origin.reshape(-1)
            tx = round(random.uniform(-10000, +10000), 2)
            ty = round(random.uniform(-10000, +10000), 2)
            tz = round(random.uniform(-10000, +10000), 2)
            target = [tx, ty, tz]
            target = np.array(target)-np.array(origin)
            c2w = viewmatrix(target, up, origin)
            tmp = np.eye(4)
            tmp[:3, :] = c2w
            c2w = tmp
            seen = check_proj(pc_unseen, W, H, fx, fy, cx, cy, c2w)
            if (~seen):
                break

        param = o3d.camera.PinholeCameraParameters()
        param.extrinsic = np.linalg.inv(c2w)  # 4x4 numpy array

        param.intrinsic = o3d.camera.PinholeCameraIntrinsic(
            W, H, fx, fy, cx, cy)

        ctr = vis.get_view_control()
        ctr.set_constant_z_far(20)
        ctr.convert_from_pinhole_camera_parameters(param)

        vis.add_geometry(gt_mesh, reset_bounding_box=True,)
        ctr.convert_from_pinhole_camera_parameters(param)
        vis.poll_events()
        vis.update_renderer()
        gt_depth = vis.capture_depth_float_buffer(True)
        gt_depth = np.asarray(gt_depth)
        vis.remove_geometry(gt_mesh, reset_bounding_box=True,)

        vis.add_geometry(rec_mesh, reset_bounding_box=True,)
        ctr.convert_from_pinhole_camera_parameters(param)
        vis.poll_events()
        vis.update_renderer()
        ours_depth = vis.capture_depth_float_buffer(True)
        ours_depth = np.asarray(ours_depth)
        vis.remove_geometry(rec_mesh, reset_bounding_box=True,)

        errors += [np.abs(gt_depth-ours_depth).mean()]

    errors = np.array(errors)
    # from m to cm
    print('Depth L1: ', errors.mean()*100)

    return {'Depth L1: ': errors.mean()*100}

def eval_rendering(cfg, n_img, frame_reader, estimate_c2w_list,
               renderer, scene_rep, decoders, truncation, output, device):

    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    from pytorch_msssim import ms_ssim
    import os
    import json
    import cv2
    import matplotlib.pyplot as plt

    # re-render frames at the end for meshing
    render_idx, frame_cnt, psnr_sum, ssim_sum, lpips_sum, depth_l1_render = 0, 0, 0, 0, 0, 0

    os.makedirs(f'{output}/rendered_image', exist_ok=True)
    os.makedirs(f'{output}/rendered_uncertainty', exist_ok=True) 

    from tqdm import tqdm
    pbar = tqdm(total=n_img, desc='Processing', unit='img')

    print('\nStart to evaluate rendering quality.')
    while render_idx < n_img:
        ret = frame_reader[render_idx]              
        idx, gt_color, gt_depth, gt_c2w, ray_d = ret
        cur_c2w = estimate_c2w_list[render_idx].to(device) 
        gt_depth = gt_depth.squeeze(0).to(device, non_blocking=True)
        gt_color = gt_color.squeeze(0).to(device, non_blocking=True)

        depth, color, termination_prob, rendered_weights, rendered_depth_uncertainty = renderer.render_img(scene_rep, decoders, cur_c2w, truncation, device, gt_depth=gt_depth)
        
        img = cv2.cvtColor(color.cpu().numpy()*255, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(f'{output}/rendered_image', f'frame_{render_idx:05d}.png'), img)
            
        rendered_depth_uncertainty_np = rendered_depth_uncertainty.cpu().numpy()
        rendered_depth_uncertainty_normalized = np.clip(rendered_depth_uncertainty_np, 0, 1.0)
        cmap = plt.get_cmap('jet')
        colored_uncertainty = cmap(rendered_depth_uncertainty_normalized)  # (H, W, 4)，RGBA
        colored_uncertainty_img = (colored_uncertainty[:, :, :3] * 255).astype('uint8')
        output_path = os.path.join(f'{output}/rendered_uncertainty', f'frame_{render_idx:05d}.png')
        cv2.imwrite(output_path, cv2.cvtColor(colored_uncertainty_img, cv2.COLOR_RGB2BGR))

        cal_lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).to(device)
        mse_loss = torch.nn.functional.mse_loss(gt_color[gt_depth > 0], color[gt_depth > 0])
        psnr_frame = -10. * torch.log10(mse_loss)
        ssim_value = ms_ssim(gt_color.transpose(0, 2).unsqueeze(0).float(), color.transpose(0, 2).unsqueeze(0).float(),
                                data_range=1.0, size_average=True)
        lpips_value = cal_lpips(torch.clamp(gt_color.unsqueeze(0).permute(0, 3, 1, 2).float(), 0.0, 1.0),
                                torch.clamp(color.unsqueeze(0).permute(0, 3, 1, 2).float(), 0.0, 1.0)).item()
        psnr_sum += psnr_frame
        ssim_sum += ssim_value
        lpips_sum += lpips_value

        depth_l1_render += torch.abs(gt_depth[gt_depth > 0] - depth[gt_depth > 0]).mean().item()
        pbar.update(cfg['mapping']['every_frame'])
        render_idx += 5
        frame_cnt += 1

    avg_psnr = psnr_sum / frame_cnt
    avg_ssim = ssim_sum / frame_cnt
    avg_lpips = lpips_sum / frame_cnt
    avg_depth_l1 = depth_l1_render/frame_cnt

    results = {"avg_ms_ssim": float(f"{avg_ssim:.4f}"),
                "avg_psnr": float(f"{avg_psnr:.4f}"),
                "avg_lpips": float(f"{avg_lpips:.4f}"),
                "depth_l1_render": float(f"{avg_depth_l1:.4f}")}

    pbar.close()
    print(results)

    filename = os.path.join(output, 'output.txt')
    with open(filename, 'a') as file:
        file.write(json.dumps(results) + '\n')
        file.write('\n')


if __name__ == '__main__':  
    parser = argparse.ArgumentParser(
        description='Arguments to evaluate the reconstruction.'
    )
    parser.add_argument('--rec_mesh', type=str,
                        help='reconstructed mesh file path')
    parser.add_argument('--gt_mesh', type=str,
                        help='ground truth mesh file path')

    parser.add_argument('-2d', '--metric_2d',
                        action='store_true', help='enable 2D metric')
    parser.add_argument('-3d', '--metric_3d',
                        action='store_true', help='enable 3D metric')
    args = parser.parse_args()
    if args.metric_3d:
        eval_3d_results = calc_3d_metric(args.rec_mesh, args.gt_mesh)

    if args.metric_2d:
        eval_2d_results = calc_2d_metric(args.rec_mesh, args.gt_mesh, n_imgs=100)

    