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
import os
import numpy as np
import numpy
import torch
import sys
import matplotlib.pyplot as plt
import json

sys.path.append('.')
from src import config
from src.common import matrix_to_cam_pose

def translation_to_linear_velocity(translations, dt = 1):
    """  
    This equation is used to calculate linear velocity and linear acceleration
    """

    linear_velocities = torch.diff(translations, dim=0)      
    initial_velocity = torch.zeros_like(linear_velocities[0])
    linear_velocities = torch.cat([initial_velocity.unsqueeze(0), linear_velocities], dim=0)
    velocity_magnitude = torch.norm(linear_velocities, dim=1)    

    velocity_acceleration = torch.diff(velocity_magnitude) / dt   
    initial_acceleration = torch.zeros_like(velocity_acceleration[0])
    velocity_acceleration = torch.cat([initial_acceleration.unsqueeze(0), velocity_acceleration], dim=0) 

    return velocity_magnitude, velocity_acceleration

def quaternion_to_rotational_velocity_round(quaternions):
    """
    Convert quaternions to rotational velocities.  
    """
    dt = 1  # assuming one frame per second

    diff_quaternions = quaternions[1:] - quaternions[:-1]
    angular_velocities = 2 * diff_quaternions / dt

    angular_velocities = torch.norm(angular_velocities, dim=1)  

    # Add initial angular velocity (zero) at the beginning
    initial_angular_velocity = torch.tensor([0], dtype=angular_velocities.dtype)
    angular_velocities = torch.cat([initial_angular_velocity, angular_velocities])

    # Angular acceleration
    diff_angular_velocity = (angular_velocities[1:] - angular_velocities[:-1])/ dt
    angular_acceleration = diff_angular_velocity

    # Add initial angular acceleration (zero) at the beginning
    initial_angular_acceleration = torch.tensor([0], dtype=angular_acceleration.dtype)
    angular_acceleration = torch.cat([initial_angular_acceleration, angular_acceleration])

    return angular_velocities, angular_acceleration

def replace_outliers_with_median(data, factor=20):
    """
    Replace outliers in the data that are greater than 'factor' times the median with the median.

    Parameters:
    data (Tensor): The input data.
    factor (float): The factor to determine outliers.

    Returns:
    Tensor: Data with outliers replaced by the median.
    """
    median = torch.median(data)
    outlier_threshold = median * factor

    # Replace values greater than outlier_threshold with the median
    replaced_data = torch.where(data > outlier_threshold, median, data)
    replaced_data = torch.where(replaced_data < -outlier_threshold, median, replaced_data)
    return replaced_data


def plot_combined_velocity(poses_gt, poses_est, rendered_weight, plot_path):
    N = poses_gt.shape[0] - 1
    directory, filename = os.path.split(plot_path)
    new_filename = filename.replace('pose', 'vel')
    plot_path = os.path.join(directory, new_filename)

    if not isinstance(poses_gt, torch.Tensor):
        poses_gt = torch.from_numpy(poses_gt)
        poses_est = torch.from_numpy(poses_est)

    translations_gt = poses_gt[:, :3]   
    quaternions_gt = poses_gt[:, 3:]
    translations_est = poses_est[:, :3]
    quaternions_est = poses_est[:, 3:]

    # Calculate linear velocity and accleration
    velocity_magnitude_gt, velocity_acceleration_gt = translation_to_linear_velocity(translations_gt)
    velocity_magnitude_est, velocity_acceleration_est = translation_to_linear_velocity(translations_est)

    # Calculate angular velocity from quaternions
    angular_velocity_gt, angular_acceleration_gt = quaternion_to_rotational_velocity_round(quaternions_gt)
    angular_velocity_est, angular_acceleration_est = quaternion_to_rotational_velocity_round(quaternions_est)

    angular_velocity_gt = replace_outliers_with_median(angular_velocity_gt)
    angular_velocity_est = replace_outliers_with_median(angular_velocity_est)
    angular_acceleration_est = replace_outliers_with_median(angular_acceleration_est)

    # Time steps for plotting
    timesteps_linear = np.arange(velocity_magnitude_gt.shape[0])
    timesteps_angular = np.arange(angular_velocity_gt.shape[0])

    # Plot linear velocity gt
    plt.figure(figsize=(12, 18))
    plt.subplot(3, 2, 1)
    plt.plot(timesteps_linear, velocity_magnitude_gt.numpy())
    plt.title('Linear Velocity GT')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (units/s)')

    # Plot angular velocity gt 
    plt.subplot(3, 2, 2)
    plt.plot(timesteps_angular, angular_velocity_gt.numpy())
    plt.title('Angular Velocity GT')
    plt.xlabel('Time (s)')
    plt.ylabel('Angular Velocity (rad/s)')

    # Plot linear velocity est
    plt.subplot(3, 2, 3)
    plt.plot(timesteps_linear, velocity_magnitude_est.numpy())
    plt.title('Linear Velocity Est')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (units/s)')

    # Plot angular velocity est
    plt.subplot(3, 2, 4)
    plt.plot(timesteps_angular, angular_velocity_est.numpy())
    plt.title('Angular Velocity Est')
    plt.xlabel('Time (s)')
    plt.ylabel('Angular Velocity (rad/s)')

    # Plot (1 - rendered weights).mean()
    plt.subplot(3, 2, 5)
    plt.plot(timesteps_angular, rendered_weight.detach().cpu().numpy())
    plt.title('uncertainty')
    plt.xlabel('Time (s)')
    plt.ylabel('uncertainty')

    # Plot (1 - rendered weights).mean()
    plt.subplot(3, 2, 6)
    plt.plot(timesteps_angular, rendered_weight.detach().cpu().numpy())
    plt.title('uncertainty')
    plt.xlabel('Time (s)')
    plt.ylabel('uncertainty')

    plt.tight_layout()
    plt.savefig(plot_path)

def associate(first_list, second_list, offset=0.0, max_difference=0.02):
    """
    Associate two dictionaries of (stamp,data). As the time stamps never match exactly, we aim
    to find the closest match for every input tuple.

    Input:
    first_list -- first dictionary of (stamp,data) tuples
    second_list -- second dictionary of (stamp,data) tuples
    offset -- time offset between both dictionaries (e.g., to model the delay between the sensors)
    max_difference -- search radius for candidate generation

    Output:
    matches -- list of matched tuples ((stamp1,data1),(stamp2,data2))

    """
    first_keys = list(first_list.keys())
    second_keys = list(second_list.keys())
    potential_matches = [(abs(a - (b + offset)), a, b)
                         for a in first_keys
                         for b in second_keys
                         if abs(a - (b + offset)) < max_difference]
    potential_matches.sort()
    matches = []
    for diff, a, b in potential_matches:
        if a in first_keys and b in second_keys:
            first_keys.remove(a)
            second_keys.remove(b)
            matches.append((a, b))

    matches.sort()
    return matches


def align(model, data):
    """Align two trajectories using the method of Horn (closed-form).

    Input:
    model -- first trajectory (3xn)
    data -- second trajectory (3xn)

    Output:
    rot -- rotation matrix (3x3)
    trans -- translation vector (3x1)
    trans_error -- translational error per point (1xn)

    """
    numpy.set_printoptions(precision=3, suppress=True)
    model_zerocentered = model - model.mean(1)
    data_zerocentered = data - data.mean(1)

    W = numpy.zeros((3, 3))
    for column in range(model.shape[1]):
        W += numpy.outer(model_zerocentered[:,
                         column], data_zerocentered[:, column])
    U, d, Vh = numpy.linalg.linalg.svd(W.transpose())
    S = numpy.matrix(numpy.identity(3))
    if(numpy.linalg.det(U) * numpy.linalg.det(Vh) < 0):
        S[2, 2] = -1
    rot = U*S*Vh
    trans = data.mean(1) - rot * model.mean(1)

    model_aligned = rot * model + trans
    alignment_error = model_aligned - data

    trans_error = numpy.sqrt(numpy.sum(numpy.multiply(
        alignment_error, alignment_error), 0)).A[0]

    return rot, trans, trans_error


def plot_traj(ax, stamps, traj, style, color, label):
    """
    Plot a trajectory using matplotlib. 

    Input:
    ax -- the plot
    stamps -- time stamps (1xn)
    traj -- trajectory (3xn)
    style -- line style
    color -- line color
    label -- plot legend

    """
    stamps.sort()
    interval = numpy.median([s-t for s, t in zip(stamps[1:], stamps[:-1])])
    x = []
    y = []
    last = stamps[0]
    for i in range(len(stamps)):
        if stamps[i]-last < 2*interval:
            x.append(traj[i][0])
            y.append(traj[i][1])
        elif len(x) > 0:
            ax.plot(x, y, style, color=color, label=label)
            label = ""
            x = []
            y = []
        last = stamps[i]
    if len(x) > 0:
        ax.plot(x, y, style, color=color, label=label)

def pose_evaluation(gt_c2w_list, estimate_c2w_list, rendered_weights, plot_path, scale, pose_alignment):
    """
 
    """
    N = len(gt_c2w_list)-1 
    poses_gt, mask = convert_poses(gt_c2w_list, N, scale)
    poses_est, _ = convert_poses(estimate_c2w_list, N, scale)
    poses_est = poses_est[mask]
    rendered_weights = rendered_weights[mask]
    trans_error, results = evaluate(poses_gt, poses_est, rendered_weights, plot_path=plot_path, pose_alignment=pose_alignment)

    return trans_error, results

def vis_trans_error(trans_error, output, file_path):
    trans_error = list(trans_error)
    trans_error_rounded = [round(err, 4) for err in trans_error] 
    data = {"trans_error": trans_error_rounded}

    # Define the file path, you can change 'trans_error_data.json' to your preferred file name
    file_path = os.path.join(output, file_path)

    # Write the dictionary to a JSON file
    with open(file_path, 'w') as file:
        json.dump(data, file)

    #######   Code for the Color Scale Bar   #######
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    # Define the colormap
    cmap = plt.get_cmap('viridis')
    norm = Normalize(vmin=0, vmax=0.001)
    fig, ax = plt.subplots(figsize=(0.35, 4), dpi=300)
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=ax)
    cb.set_label('Uncertainty Value')        

    cb.ax.yaxis.label.set_size(10)
    cb.ax.yaxis.set_tick_params(labelsize=8)
    cb.ax.yaxis.set_tick_params(right=False)  

    # Save the colorbar figure
    fig.savefig(os.path.join(output, 'uncertainty_colorbar.png'), bbox_inches='tight')
    print("Uncertainty colorbar is saved in {}".format(os.path.join(output, 'uncertainty_colorbar.png')))

    #######  the scatter plot image  #######
    cmap = plt.get_cmap('jet')
    norm = plt.Normalize(vmin=0, vmax=0.25)
    fig, ax = plt.subplots(figsize=(6, 0.5), dpi=300)

    for i, value in enumerate(trans_error):
        ax.vlines(i, 0, 1, color=cmap(norm(value)), linewidth=2)

    ax.yaxis.set_visible(False)
    # Set the x-axis limits to match the number of frames
    ax.set_xticks(range(0, len(trans_error), 500))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_aspect('auto')
    fig.savefig(os.path.join(output, 'translation_error_scatter.png'), bbox_inches='tight', pad_inches=0)
    print("translation_error is saved in {}".format(os.path.join(output,'translation_error_scatter.png')))


def vis_unc_mapstep(tracking_rendered_weight_list, addtional_map_records, output):
    """
    Args:
        tracking_rendered_weight_list: Uncertainty values per frame.
        addtional_map_records: 0/1 flags indicating additional mapping activation.
        output: Directory to save output images.
    """

    cmap = plt.get_cmap('plasma')
    norm = plt.Normalize(vmin=0, vmax=0.005)
    fig, ax = plt.subplots(figsize=(6, 0.5), dpi=300)

    for i, value in enumerate(tracking_rendered_weight_list):
        ax.vlines(i, 0, 1, color=cmap(norm(value)), linewidth=2)

    ax.yaxis.set_visible(False)
    ax.set_xticks(range(0, len(tracking_rendered_weight_list), 500))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_aspect('auto')
    fig.savefig(os.path.join(output, 'uncertainty_record.png'), bbox_inches='tight', pad_inches=0)

    #######  the map scatter plot image  #######
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(['white', 'red'])
    fig, ax = plt.subplots(figsize=(6, 0.5), dpi=300)

    for i, value in enumerate(addtional_map_records):
        color = 'white' if value == 0 else 'red'
        ax.vlines(i, 0, 1, color=color, linewidth=2)

    ax.yaxis.set_visible(False)
    ax.set_xticks(range(0, len(addtional_map_records), 500))
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    fig.savefig(os.path.join(output, 'addtional_mapping_record.png'), bbox_inches='tight', pad_inches=0)

def evaluate_ate(first_list, second_list, plot="", _args="", pose_alignment=False):
    
    parser = argparse.ArgumentParser(
        description='This script computes the absolute trajectory error from the ground truth trajectory and the estimated trajectory.')
    parser.add_argument(
        '--offset', help='time offset added to the timestamps of the second file (default: 0.0)', default=0.0)
    parser.add_argument(
        '--scale', help='scaling factor for the second trajectory (default: 1.0)', default=1.0)
    parser.add_argument(
        '--max_difference', help='maximally allowed time difference for matching entries (default: 0.02)', default=0.02)
    parser.add_argument(
        '--save', help='save aligned second trajectory to disk (format: stamp2 x2 y2 z2)')
    parser.add_argument('--save_associations',
                        help='save associated first and aligned second trajectory to disk (format: stamp1 x1 y1 z1 stamp2 x2 y2 z2)')
    parser.add_argument(
        '--plot', help='plot the first and the aligned second trajectory to an image (format: png)')
    parser.add_argument(
        '--verbose', help='print all evaluation data (otherwise, only the RMSE absolute translational error in meters after alignment will be printed)', action='store_true')
    args = parser.parse_args(_args)
    args.plot = plot

    matches = associate(first_list, second_list, float(
        args.offset), float(args.max_difference))
    if len(matches) < 2:
        raise ValueError(
            "Couldn't find matching timestamp pairs between groundtruth and estimated trajectory! \
            Did you choose the correct sequence?")

    first_xyz = numpy.matrix(
        [[float(value) for value in first_list[a][0:3]] for a, b in matches]).transpose()
    second_xyz = numpy.matrix([[float(value)*float(args.scale)
                              for value in second_list[b][0:3]] for a, b in matches]).transpose()

    rot, trans, trans_error = align(second_xyz, first_xyz)
    output = os.path.dirname(plot)

    vis_trans_error(trans_error, output, 'trans_error_data.json')

    if pose_alignment:
        second_xyz_aligned = rot * second_xyz + trans
    else:
        second_xyz_aligned = second_xyz

    first_stamps = list(first_list.keys())
    first_stamps.sort()
    first_xyz_full = numpy.matrix(
        [[float(value) for value in first_list[b][0:3]] for b in first_stamps]).transpose()

    second_stamps = list(second_list.keys())
    second_stamps.sort()
    second_xyz_full = numpy.matrix([[float(value)*float(args.scale)
                                   for value in second_list[b][0:3]] for b in second_stamps]).transpose()
    
    if pose_alignment:
        second_xyz_full_aligned = rot * second_xyz_full + trans    
    else:
        second_xyz_full_aligned = second_xyz_full      

    args.verbose = True
    if args.verbose:
        print("compared_pose_pairs %d pairs" % (len(trans_error)))

        print("absolute_translational_error.rmse %f m" % numpy.sqrt(
            numpy.dot(trans_error, trans_error) / len(trans_error)))
        print("absolute_translational_error.mean %f m" %
              numpy.mean(trans_error))
        print("absolute_translational_error.median %f m" %
              numpy.median(trans_error))
        print("absolute_translational_error.std %f m" % numpy.std(trans_error))
        print("absolute_translational_error.min %f m" % numpy.min(trans_error))
        print("absolute_translational_error.max %f m" % numpy.max(trans_error))

    if args.save_associations:
        file = open(args.save_associations, "w")
        file.write("\n".join(["%f %f %f %f %f %f %f %f" % (a, x1, y1, z1, b, x2, y2, z2) for (
            a, b), (x1, y1, z1), (x2, y2, z2) in zip(matches, first_xyz.transpose().A, second_xyz_aligned.transpose().A)]))
        file.close()

    if args.save:
        file = open(args.save, "w")
        file.write("\n".join(["%f " % stamp+" ".join(["%f" % d for d in line])
                   for stamp, line in zip(second_stamps, second_xyz_full_aligned.transpose().A)]))
        file.close()

    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pylab as pylab
        import matplotlib.pyplot as plt
        from matplotlib.patches import Ellipse
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ATE = numpy.sqrt(
            numpy.dot(trans_error, trans_error) / len(trans_error))
        ax.set_title(f'len:{len(trans_error)} ATE RMSE:{ATE} {args.plot[:-3]}')
        plot_traj(ax, first_stamps, first_xyz_full.transpose().A,
                  '-', "black", "ground truth")
        plot_traj(ax, second_stamps, second_xyz_full_aligned.transpose(
        ).A, '-', "blue", "estimated")

        # Mark the end point of the ground truth trajectory
        last_gt_x, last_gt_y = first_xyz_full[:,-1][:2]  
        ax.plot(last_gt_x, last_gt_y, 'o', color="green", markersize=10, label="GT End Point")

        # Mark the end point of the estimated trajectory
        last_est_x, last_est_y = second_xyz_full_aligned[:,-1][:2]  
        ax.plot(last_est_x, last_est_y, '*', color="red", markersize=10, label="Est. End Point")

        label = "difference"
        for (a, b), (x1, y1, z1), (x2, y2, z2) in zip(matches, first_xyz.transpose().A, second_xyz_aligned.transpose().A):
            label = ""
        ax.legend()
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        plt.savefig(args.plot, dpi=90)

        trans_error = trans_error*100 

    return trans_error, {
        "compared_pose_pairs": (len(trans_error)),
        "unit": "cm",
        "error.rmse": round(numpy.sqrt(numpy.dot(trans_error, trans_error) / len(trans_error)), 2),
        "error.mean": round(numpy.mean(trans_error), 2),
        "error.median": round(numpy.median(trans_error), 2),
        "error.std": round(numpy.std(trans_error), 2),
        "error.max": round(numpy.max(trans_error), 2),
    }

def evaluate(poses_gt, poses_est, rendered_weight, plot_path, pose_alignment):

    poses_gt = poses_gt.cpu().numpy()
    poses_est = poses_est.cpu().numpy()

    if rendered_weight == None:
        pass
    else:
        plot_combined_velocity(poses_gt, poses_est, rendered_weight, plot_path)

    N = poses_gt.shape[0]
    poses_gt = dict([(i, poses_gt[i]) for i in range(N)])
    poses_est = dict([(i, poses_est[i]) for i in range(N)])

    trans_error, results = evaluate_ate(poses_gt, poses_est, plot_path, pose_alignment=pose_alignment)
    print(results)

    return trans_error, results

def convert_poses(c2w_list, N, scale, gt=True):
    poses = []
    mask = torch.ones(N+1).bool()
    for idx in range(0, N+1):
        if gt:
            # some frame have `nan` or `inf` in gt pose of ScanNet, 
            # but our system have estimated camera pose for all frames
            # therefore, when calculating the pose error, we need to mask out invalid pose
            if torch.isinf(c2w_list[idx]).any():
                mask[idx] = 0
                continue
            if torch.isnan(c2w_list[idx]).any():
                mask[idx] = 0
                continue
        c2w_list[idx][:3, 3] /= scale
        poses.append(matrix_to_cam_pose(c2w_list[idx].unsqueeze(0), RT=False))

    poses = torch.cat(poses, dim=0)

    return poses, mask

if __name__ == '__main__':
    """
    This ATE evaluation code is modified upon the evaluation code in lie-torch.
    """
    parser = argparse.ArgumentParser(
        description='Arguments to eval the tracking ATE.'
    )
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--output', type=str,
                        help='output folder, this have higher priority, can overwrite the one inconfig file')

    args = parser.parse_args()
    cfg = config.load_config(args.config, 'configs/UNISLAM.yaml')
    scale = cfg['scale']
    output = cfg['data']['output'] if args.output is None else args.output
    ckptsdir = f'{output}/ckpts'
    if os.path.exists(ckptsdir):
        ckpts = [os.path.join(ckptsdir, f)
                 for f in sorted(os.listdir(ckptsdir)) if 'tar' in f]
        if len(ckpts) > 0:
            ckpt_path = ckpts[-1]
            print('Get ckpt :', ckpt_path)
            ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
            estimate_c2w_list = ckpt['estimate_c2w_list']
            gt_c2w_list = ckpt['gt_c2w_list']
            N = ckpt['idx']
            poses_gt, mask = convert_poses(gt_c2w_list, N, scale)
            poses_est, _ = convert_poses(estimate_c2w_list, N, scale)
            poses_est = poses_est[mask]
            evaluate(poses_gt, poses_est, rendered_weight = None, plot_path=f'{output}/eval_ate_plot.png', pose_alignment=False)
            
            tracking_rendered_weight_list = ckpt['tracking_rendered_weight_list']
            addtional_map_records = ckpt['addtional_map_records']
            vis_unc_mapstep(tracking_rendered_weight_list, addtional_map_records, output)

