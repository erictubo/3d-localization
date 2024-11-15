"""
Data conversion utilities for GLACE scene coordinate regression.
Handles conversion from CAD renders or SfM reconstructions to GLACE-compatible format.
"""
from pathlib import Path
from typing import List
import numpy as np
import cv2 as cv
import torch

from colmap_model import ColmapModelReader
from model_conversion import *

"""
REFERENCE:
`generate_from_reconstruction` function adapted and generalized from GLACE's `datasets/setup_cambridge.py` script.
"""

class GlaceConversion:
    """
    Conversion of data for compatibility with GLACE Scene Coordinate Regression:
    - Read from (A) CAD renders or (B) SFM reconstruction
        - images
        - intrinsics
        - poses
        - depth maps
        - scene coordinates (SFM only)
    - Write to train/*, test/*
    """

    def __init__(self, path_to_glace: Path, num_test: int = 0, target_height: int = 480):

        self.path_to_glace = path_to_glace
        self.num_test = num_test
        self.target_height = target_height


    def generate_from_reconstruction(
            self,
            path_to_nvm: Path,
            path_to_images: Path,
            T_sfm_cad: np.ndarray,
            depth_maps: bool = True,
            scene_coordinates: bool = True,
            to_mm = True,
            nn_subsampling = 8,
        ):
        """
        Generate GLACE train and test data from NVM file.
        """

        names = [image.name for image in path_to_images.glob('*')]
        names = [name.split('.')[0] for name in names]
        names.sort()

        n = self.num_test
        if n != 0: print(f'Names (test split): {names[:3]} ...')
        print(f'Names (train split): {names[n:n+3]} ...')


        # existing = 0
        subpaths = ['rgb', 'calibration', 'poses']
        if depth_maps: subpaths.append('depth')
        if scene_coordinates: subpaths.append('init')
        
        # Create directories
        for split in ['train', 'test']:
            path = self.path_to_glace / split
            if not path.exists(): path.mkdir()

            # total_split = self.num_test if split=='test' else len(names)-self.num_test

            for subpath in subpaths:
                path = self.path_to_glace / split / subpath
                if not path.exists(): path.mkdir()
                # else: existing += int(len([path.glob('*')]) >= total_split)
            
        # if existing == 2*len(subpaths):
        #     print('All files already exist ... skipping conversion.')
        #     return


        # Set up frame conversion
        conversion = ModelConversion(T_sfm_cad=T_sfm_cad)
        T_cad_sfm = np.linalg.inv(T_sfm_cad)

        T_sfm_cad = torch.tensor(T_sfm_cad).float()
        T_cad_sfm = torch.tensor(T_cad_sfm).float()

        scale = conversion.s_cad_sfm
        print(f'Scale: {scale}')


        print("Loading SfM reconstruction...")
        file = open(path_to_nvm, 'r')
        reconstruction = file.readlines()
        file.close()

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

        for cam_idx in range(num_cams):

            # Read data from reconstruction file
            line = reconstruction[3 + cam_idx].split()
            image_name = line[0]
            focal_length = float(line[1])

            name = image_name.split('.')[0]

            if self.num_test > 0 and names.index(name) < self.num_test:
                split = 'test'
            else:
                split = 'train'

            print(f"{cam_idx + 1} / {num_cams}: {image_name} -> {split}")

            pose_file = self.path_to_glace / split / 'poses' / f'{name}.txt'
            image_file = self.path_to_glace / split / 'rgb' / image_name
            intrinsics_file = self.path_to_glace / split / 'calibration' / f'{name}.txt'
            depth_file = self.path_to_glace / split / 'depth' / f'{name}.npy'
            coords_file = self.path_to_glace / split / 'init' / f'{name}.dat'

            out_files = [pose_file, image_file, intrinsics_file]
            if depth_maps: out_files.append(depth_file)
            if scene_coordinates: out_files.append(coords_file)

            file_existing = 0
            for file in out_files:
                file_existing += int(file.exists())
            if file_existing == len(out_files):
                print(f'File {name} already exist ... skipping conversion.')
                continue


            # POSE
            t_sfm_cam = np.asarray([float(r) for r in line[6:9]])   # camera center in SfM coordinate system

            q_cam_sfm = np.asarray([float(r) for r in line[2:6]])   # camera rotation in CAM frame

            R_cam_sfm = Quaternion(q_cam_sfm).rotation_matrix
            R_sfm_cam = R_cam_sfm.T

            T_sfm_cam = np.eye(4)
            T_sfm_cam[:3, :3] = R_sfm_cam
            T_sfm_cam[:3, 3] = t_sfm_cam

            T_cam_sfm = np.linalg.inv(T_sfm_cam)

            T_cad_cam = conversion.transform_pose_from_colmap_to_cad_format(T_cam_sfm, to_blender_format=False)
            pose_cad_cam = convert_matrix_to_pose(T_cad_cam)

            np.savetxt(pose_file, T_cad_cam, fmt='%15.7e')


            T_cam_sfm = torch.tensor(T_cam_sfm).float()
            T_sfm_cam = torch.tensor(T_sfm_cam).float()


            # IMAGE
            image = cv.imread(path_to_images / image_name)

            img_aspect = image.shape[0] / image.shape[1]

            if img_aspect > 1:
                img_w = self.target_height
                img_h = int(np.ceil(self.target_height * img_aspect))
            else:
                img_w = int(np.ceil(self.target_height / img_aspect))
                img_h = self.target_height

            out_w = int(np.ceil(img_w / nn_subsampling))
            out_h = int(np.ceil(img_h / nn_subsampling))

            out_scale = out_w / image.shape[1]
            img_scale = img_w / image.shape[1]

            image = cv.resize(image, (img_w, img_h))
            cv.imwrite(image_file, image)


            # INTRINSICS
            with open(intrinsics_file, 'w') as f:
                f.write(str(focal_length * img_scale))


            # DEPTH MAPS & SCENE COORDINATES
            if depth_maps or scene_coordinates:

                # load 3D points from reconstruction
                pts_3D = torch.tensor(pts_dict[cam_idx])

                depth_sfm = torch.zeros((out_h, out_w))
                coords_sfm = torch.zeros((3, out_h, out_w))

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

                    x = int(torch.clamp(x, min=0, max=coords_sfm.size(2) - 1))
                    y = int(torch.clamp(y, min=0, max=coords_sfm.size(1) - 1))

                    if cam_pt[2, 0] > 1000:  # filter some outlier points (large depth)
                        continue

                    if depth_sfm[y, x] == 0 or depth_sfm[y, x] > cam_pt[2, 0]:
                        depth_sfm[y, x] = cam_pt[2, 0]
                        coords_sfm[:, y, x] = pts_3D[pt_idx, 0:3]


                if depth_maps:
                    depth_sfm = depth_sfm.numpy()
                    assert depth_sfm.shape == (out_h, out_w), depth_sfm.shape

                    if to_mm:
                        depth_sfm *= 1000

                    depth_cad = depth_sfm * scale

                    np.save(depth_file, depth_cad)
                
                if scene_coordinates:

                    valid_mask = coords_sfm.sum(dim=0) != 0

                    coords_sfm_valid = coords_sfm[:, valid_mask]
                    coords_sfm_valid_homogeneous = torch.cat([coords_sfm_valid, torch.ones(1, coords_sfm_valid.size(1))], dim=0)

                    coords_cad_valid_homogeneous = torch.mm(T_cad_sfm, coords_sfm_valid_homogeneous)
                    coords_cad_valid = coords_cad_valid_homogeneous[0:3, :]

                    coords_cad = torch.zeros((3, out_h, out_w))
                    coords_cad[:, valid_mask] = coords_cad_valid

                    torch.save(coords_cad, coords_file)


    def _generate_from_renders(
            self,
            path_to_renders: Path,
            depth_maps: bool = True,
            to_mm = True,
            nn_subsampling = 1,
        ):
        """
        Generate GLACE train and test data from renders.
        """

        names = [image.name for image in path_to_renders.glob('images/*')]
        names = [name.split('.')[0] for name in names]
        names.sort()

        n = self.num_test
        if n != 0: print(f'Names (test split): {names[:3]} ...')
        print(f'Names (train split): {names[n:n+3]} ...')


        # existing = 0
        subpaths = ['rgb', 'calibration', 'poses']
        if depth_maps: subpaths.append('depth')
        
        # Create directories
        for split in ['train', 'test']:
            path = self.path_to_glace / split
            if not path.exists(): path.mkdir()

            # total_split = self.num_test if split=='test' else len(names)-self.num_test

            for subpath in subpaths:
                path = self.path_to_glace / split / subpath
                if not path.exists(): path.mkdir()
                # else: existing += int(len([path.glob('*')]) >= total_split)
            
        # if existing == 2*len(subpaths):
        #     print('All files already exist ... skipping conversion.')
        #     return
        

        total = len(names)
        for i, name in enumerate(names):

            if i < self.num_test: split = 'test'
            else: split = 'train'

            print(f"{i + 1} / {total}: {name} -> {split}")

            pose_file = self.path_to_glace / split / 'poses' / f'{name}.txt'
            image_file = self.path_to_glace / split / 'rgb' / f'{name}.png'
            intrinsics_file = self.path_to_glace / split / 'calibration' / f'{name}.txt'
            depth_file = self.path_to_glace / split / 'depth' / f'{name}.npy'

            out_files = [pose_file, image_file, intrinsics_file]
            if depth_maps: out_files.append(depth_file)

            file_existing = 0
            for file in out_files:
                file_existing += int(file.exists())
            if file_existing == len(out_files):
                print(f'File {name} already exist ... skipping conversion.')
                continue


            # POSE
            pose_cad_cam_blender = np.loadtxt(path_to_renders / 'poses' / f'{name}.txt')
            T_cad_cam_blender = convert_pose_to_matrix(pose_cad_cam_blender)

            T_cad_cam = reverse_camera_pose_for_blender(T_cad_cam_blender,'CAD')

            np.savetxt(pose_file, T_cad_cam, fmt='%15.7e')


            # IMAGE
            image = cv.imread(path_to_renders / 'images' / f'{name}.png')

            img_aspect = image.shape[0] / image.shape[1]

            if img_aspect > 1:
                img_w = self.target_height
                img_h = int(np.ceil(self.target_height * img_aspect))
            else:
                img_w = int(np.ceil(self.target_height / img_aspect))
                img_h = self.target_height

            out_w = int(np.ceil(img_w / nn_subsampling))
            out_h = int(np.ceil(img_h / nn_subsampling))

            out_scale = out_w / image.shape[1]
            img_scale = img_w / image.shape[1]
            
            image = cv.resize(image, (img_w, img_h))
            cv.imwrite(image_file, image)


            # INTRINSICS
            file = path_to_renders / 'intrinsics' / f'{name}.txt'

            w, h, f, f_unit, _, _ = file.read_text().strip().split()
            w, h, f = int(w), int(h), float(f)

            if f_unit == 'MM':
                # convert focal length from mm to pixels
                focal_length = f * max(w, h) / 36
                focal_length *= img_scale
            else:
                raise NotImplementedError(f"Unit {f_unit} not implemented")
            
            with open(intrinsics_file, 'w') as f:
                f.write(str(focal_length * img_scale))


            # DEPTH MAPS
            if depth_maps:
                    
                depth_map = ModelConversion.convert_depth_map_from_exr_to_numpy(path_to_renders / 'depth/', name)

                # resize depth map & subsample by using nearest neighbor
                depth_map_subsampled = np.zeros((out_h, out_w))

                scale_x = out_w / depth_map.shape[1]
                scale_y = out_h / depth_map.shape[0]

                for y in range(out_h):
                    for x in range(out_w):
                        x_ = int( (x+0.5) / scale_x)
                        y_ = int( (y+0.5) / scale_y)

                        depth_map_subsampled[y, x] = depth_map[y_, x_]

                #         if x==0 and y==0: print(f'({x}, {y}) <- ({x_}, {y_})')
                # print(f'({x}, {y}) <- ({x_}, {y_})')
                
                depth_cad = depth_map_subsampled
                assert depth_cad.shape == (out_h, out_w), depth_cad.shape

                if to_mm:
                    depth_map_subsampled *= 1000

                np.save(depth_file, depth_map_subsampled)


if __name__ == '__main__':

    from data import path_to_data

    T_notre_dame_B = np.array([
        [-0.04308, -0.07366, -0.0008805, -1.525],
        [0.0245, -0.01336, -0.08065, 4.145],
        [0.06947, -0.04097, 0.02789, 10.74],
        [0, 0, 0, 1],
    ])

    T_pantheon_B = np.array([
        [-0.1956, 0.005829, -0.0004737, -0.05305],
        [-0.001383, -0.06151, -0.1857, 5.694],
        [-0.005682, -0.1856, 0.06152, 13.51],
        [0, 0, 0, 1],
    ])

    T_brandenburg_gate_B = np.array([
        [-0.02426, 0.1181, 0.003819, 0.9054],
        [-0.03036, -0.002461, -0.1167, 1.876],
        [-0.1142, -0.02443, 0.03022, 7.294],
        [0, 0, 0, 1],
    ])

    T_reichstag_A = np.array([
        [-0.0004522, -0.06534, 0.0007033, 0.4911],
        [0.00916, -0.0007597, -0.06469, 1.748],
        [0.06469, -0.0003491, 0.009164, 12.37],
        [0, 0, 0, 1],
    ])

    T_ref = {
        'notre_dame': T_notre_dame_B,
        'pantheon': T_pantheon_B,
        'brandenburg_gate': T_brandenburg_gate_B,
        'reichstag': T_reichstag_A,
    }

    cad_models = {
        'notre_dame': ['notre_dame_B'],
        'pantheon': ['pantheon_B'],
        'brandenburg_gate': ['brandenburg_gate_B'],
        'reichstag': ['reichstag_A'],
    }


    for model in ['notre_dame', 'pantheon', 'brandenburg_gate', 'reichstag']:

        name = model.title().replace('_', ' ')
        path_to_colmap_model = path_to_data / f'3D Models/{name}/Reference/dense/sparse'

        print(f'Reference model: {name}')

        GlaceConversion(
            path_to_glace=path_to_data / f'GLACE/{model}',
            num_test=100,
        ).generate_from_reconstruction(
            path_to_nvm=path_to_colmap_model / 'reconstruction.nvm',
            path_to_images=path_to_colmap_model.parent / 'images',
            T_sfm_cad=T_ref[model],
        )


        for cad_model in cad_models[model]:

            cad_name = cad_model.replace('_', ' ')
            path_to_renders = path_to_data / f'Evaluation/{cad_name}/ground truth/renders'

            print(f'- CAD Model: {cad_name}')

            GlaceConversion(
                path_to_glace=path_to_data / f'GLACE/{cad_model}',
                num_test=100,
            )._generate_from_renders(
                path_to_renders=path_to_renders,
            )

            # TODO: for models not using the main T_ref, transfer back and forth to same frame
            # e.g. pantheon_C: transform to coordinates of pantheon_B
            # coordinates_B = T_B_C * coordinates_C
            # T_B_C = T_B_sfm * T_sfm_C = (T_sfm_B)^-1 * T_sfm_C


    # Pantheon

    # path_to_renders = path_to_data / 'Evaluation/pantheon B/ground truth/renders'

    # path_to_colmap_model = path_to_data / '3D Models/Pantheon/Reference/dense/sparse'
    # path_to_nvm = path_to_colmap_model / 'reconstruction.nvm'
    # path_to_images = path_to_colmap_model.parent / 'images'

    # GlaceConversion(
    #     path_to_glace=path_to_data / f'GLACE/pantheon (SFM)',
    #     num_test=100,
    # ).generate_from_reconstruction(
    #     path_to_nvm=path_to_nvm,
    #     path_to_images=path_to_images,
    #     T_sfm_cad=T_pantheon,
    # )

    # GlaceConversion(
    #     path_to_glace=path_to_data / f'GLACE/pantheon_B',
    #     num_test=100,
    # )._generate_from_renders(
    #     path_to_renders=path_to_renders,
    # )