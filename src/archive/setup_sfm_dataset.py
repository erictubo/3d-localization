#!/usr/bin/env python3

import argparse
import math
import os

import cv2 as cv
from pyquaternion import Quaternion
import numpy as np
import torch

from model_conversion import *

# NVM_V3 File Structure, see http://ccwu.me/vsfm/doc.html#nvm
# NVM_V3 [optional calibration]                        # file version header
# <Model1> <Model2> ...                                # multiple reconstructed models
# <Empty Model containing the unregistered Images>     # number of camera > 0, but number of points = 0
# <0>                                                  # 0 camera to indicate the end of model section
# <Some comments describing the PLY section>
# <Number of PLY files> <List of indices of models that have associated PLY>

# Each reconstructed <model> contains the following
# <Number of cameras>   <List of cameras>
# <Number of 3D points> <List of points>

# The cameras and 3D points are saved in the following format
# <Camera> = <File name> <focal length> <quaternion WXYZ> <camera center> <radial distortion> 0
# <Point>  = <XYZ> <RGB> <number of measurements> <List of Measurements>
# <Measurement> = <Image index> <Feature Index> <xy>


# Input file structure:
# - reconstruction.nvm: SfM reconstruction exported from COLMAP
# - /images*: folder with images
# - cam_sfm_poses.txt: camera poses in SfM coordinate system
# - T_sfm_cad.txt: transformation from SfM to realistic CAD coordinate system

# TODO: make into a function
# TODO: add option to export depth maps without subsampling, rather than scene coordinates
# export to /depth/ folder as .npy files in mm

num_test = 100  # number of test images
target_height = 480  # rescale images
nn_subsampling = 8  # sub sampling of our CNN architecture, for size of the initalization targets

def mkdir(directory):
	"""Checks whether the directory exists and creates it if necessacy."""
	if not os.path.exists(directory):
		os.makedirs(directory)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Setup SFM dataset for GLACE.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('scene_dir', type=str)

    parser.add_argument('--init', type=str, choices=['none', 'sfm'], default='none',
                        help='none: no initialisation targets for scene coordinates; sfm: scene coordinate targets by rendering the SfM point cloud')

    opt = parser.parse_args()

    input_file = 'reconstruction.nvm'

    print("Loading SfM reconstruction...")

    os.chdir(opt.scene_dir)

    f = open(input_file)
    reconstruction = f.readlines()
    f.close()

    num_cams = int(reconstruction[2])
    num_pts = int(reconstruction[num_cams + 4])

    # read points
    pts_dict = {}
    for cam_idx in range(0, num_cams):
        pts_dict[cam_idx] = []

    pt = pts_start = num_cams + 5
    pts_end = pts_start + num_pts

    while pt < pts_end:

        pt_list = reconstruction[pt].split()
        pt_3D = [float(x) for x in pt_list[0:3]]
        pt_3D.append(1.0)

        for pt_view in range(0, int(pt_list[6])):
            cam_view = int(pt_list[7 + pt_view * 4])
            pts_dict[cam_view].append(pt_3D)

        pt += 1

    print("Reconstruction contains %d cameras and %d 3D points." % (num_cams, num_pts))

    for split in ['train', 'test']:
        mkdir(split)
        mkdir(split + '/rgb/')
        mkdir(split + '/calibration/')
        mkdir(split + '/poses/')
        mkdir(split + '/init/')

    # Load transformation from SfM to CAD coordinate system
    T_sfm_cad = np.loadtxt('T_sfm_cad.txt')
    assert T_sfm_cad.shape == (4, 4)
    T_cad_sfm = np.linalg.inv(T_sfm_cad)

    T_sfm_cad = torch.tensor(T_sfm_cad).float()
    T_cad_sfm = torch.tensor(T_cad_sfm).float()

    conversion = ModelConversion(Path(opt.scene_dir))


    for cam_idx in range(num_cams):

        if num_test > 0 and cam_idx < num_test:
            split = 'test'
        else:
            split = 'train'

        # Read data from reconstruction file
        line = reconstruction[3 + cam_idx].split()
        image_file = line[0]
        focal_length = float(line[1])

        print(f"{cam_idx + 1} / {num_cams}: {image_file} -> {split}")

        t_sfm_cam = np.asarray([float(r) for r in line[6:9]])   # camera center in SfM coordinate system

        q_cam_sfm = np.asarray([float(r) for r in line[2:6]])   # camera rotation in CAM frame

        R_cam_sfm = Quaternion(q_cam_sfm).rotation_matrix
        R_sfm_cam = R_cam_sfm.T

        T_sfm_cam = np.eye(4)
        T_sfm_cam[:3, :3] = R_sfm_cam
        T_sfm_cam[:3, 3] = t_sfm_cam

        T_cam_sfm = np.linalg.inv(T_sfm_cam)


        # POSE
        T_cad_cam = conversion.transform_pose_from_colmap_to_cad_format(T_cam_sfm, to_blender_format=False)
        pose_cad_cam = convert_matrix_to_pose(T_cad_cam)

        np.savetxt(split + '/poses/' + image_file[:-3] + 'txt', T_cad_cam, fmt='%15.7e')

        T_cam_sfm = torch.tensor(T_cam_sfm).float()
        T_sfm_cam = torch.tensor(T_sfm_cam).float()


        # IMAGE
        image = cv.imread('images/' + image_file)

        img_aspect = image.shape[0] / image.shape[1]

        if img_aspect > 1:
            img_w = target_height
            img_h = int(math.ceil(target_height * img_aspect))
        else:
            img_w = int(math.ceil(target_height / img_aspect))
            img_h = target_height

        out_w = int(math.ceil(img_w / nn_subsampling))
        out_h = int(math.ceil(img_h / nn_subsampling))

        out_scale = out_w / image.shape[1]
        img_scale = img_w / image.shape[1]

        image = cv.resize(image, (img_w, img_h))
        
        cv.imwrite(split + '/rgb/' + image_file, image)


        # INTRINSICS
        with open(split + '/calibration/' + image_file[:-3] + 'txt', 'w') as f:
            f.write(str(focal_length * img_scale))


        # SCENE COORDINATES

        # load 3D points from reconstruction
        pts_3D = torch.tensor(pts_dict[cam_idx])

        out_tensor = torch.zeros((3, out_h, out_w))
        out_zbuffer = torch.zeros((out_h, out_w))

        fine = 0
        conflict = 0

        for pt_idx in range(0, pts_3D.size(0)):

            scene_pt = pts_3D[pt_idx]
            scene_pt = scene_pt.unsqueeze(0)
            scene_pt = scene_pt.transpose(0, 1)

            # scene to camera coordinates
            cam_pt = torch.mm(T_cam_sfm, scene_pt)
            # projection to image
            img_pt = cam_pt[0:2, 0] * focal_length / cam_pt[2, 0] * out_scale

            y = img_pt[1] + out_h / 2
            x = img_pt[0] + out_w / 2

            x = int(torch.clamp(x, min=0, max=out_tensor.size(2) - 1))
            y = int(torch.clamp(y, min=0, max=out_tensor.size(1) - 1))

            if cam_pt[2, 0] > 1000:  # filter some outlier points (large depth)
                continue

            if out_zbuffer[y, x] == 0 or out_zbuffer[y, x] > cam_pt[2, 0]:
                out_zbuffer[y, x] = cam_pt[2, 0]
                out_tensor[:, y, x] = pts_3D[pt_idx, 0:3]

        # create valid mask wher out_tensor is not zero
        valid_mask = out_tensor.sum(dim=0) != 0

        valid_points = out_tensor[:, valid_mask]
        valid_points_homogeneous = torch.cat([valid_points, torch.ones(1, valid_points.size(1))], dim=0)

        transformed_points = torch.mm(T_cad_sfm, valid_points_homogeneous)
        transformed_points = transformed_points[0:3, :]

        out_tensor_transformed = torch.zeros((3, out_h, out_w))
        out_tensor_transformed[:, valid_mask] = transformed_points


        # # add homogeneous coordinate
        # out_tensor = torch.cat([out_tensor, torch.ones(1, out_h, out_w)], dim=0)
        # # transform each 3D point to CAD frame
        # out_tensor = torch.mm(T_cad_sfm, out_tensor.view(4, -1)).view(4, out_h, out_w)
        # # remove homogeneous coordinate
        # out_tensor = out_tensor[0:3, :, :]

        torch.save(out_tensor_transformed, split + '/init/' + image_file[:-4] + '.dat')
