from pathlib import Path
from typing import List
import numpy as np
import cv2 as cv
import torch

from colmap_model import ColmapModelReader
from model_conversion import *


class GlaceConversion:
    """
    Conversion of data for compatibility with GLACE
    - Read from path_to_renders/* and path_to_colmap_model/* (using COLMAP data)
    - Write to train/*, test/*
    """

    def __init__(
            self,
            source: str,
            path_to_glace: Path,
            path_to_renders: Path = None,
            path_to_colmap_model: Path = None,
            T_sfm_cad: np.ndarray = None,
            num_test: int = 0,
            path_to_nvm: Path = None,
            target_height: int = 480,
        ):

        self.path_to_glace = path_to_glace
        for split in ['train', 'test']:
            path = self.path_to_glace / split
            if not path.exists(): path.mkdir()

        self.num_test = num_test

        self.target_height = target_height

        if source.lower() in ['renders', 'cad']:
            assert path_to_renders, "Missing input: path_to_renders"
            self.path_to_renders = path_to_renders

            names = [image.name for image in self.path_to_renders.glob('images/*')]
            names = [name.split('.')[0] for name in names]
            names.sort()

            # self._copy_rendered_images(names)
            # self._convert_rendered_intrinsics(names)
            # self._convert_rendered_poses(names)

            # if depth_maps:
            #     self._convert_rendered_depth_maps(names)

            self._generate_from_renders(names)

        
        elif source.lower() in ['reconstruction', 'sfm']:
            assert path_to_colmap_model, "Missing input: path_to_colmap_model"
            self.path_to_colmap_model = path_to_colmap_model

            assert T_sfm_cad is not None, "Missing input: reference transformation T_sfm_cad"
            assert T_sfm_cad.shape == (4,4), "Reference transformation is not a 4x4 matrix"
            self.T_sfm_cad = T_sfm_cad
            self.T_cad_sfm = np.linalg.inv(self.T_sfm_cad)

            # self.scale = np.linalg.norm(self.T_sfm_cad[:3, 3])
            # print(f'Scale: {round(self.scale, 2)}')
            # assert self.scale > 1, "Invalid scale"

            self.path_to_images = self.path_to_colmap_model.parent / 'images'

            sfm_names = [image.name for image in self.path_to_images.glob('*')]
            sfm_names = [name.split('.')[0] for name in sfm_names]
            sfm_names.sort()

            n = self.num_test
            if n != 0: print(f'SFM names (test split): {sfm_names[:3]}')
            print(f'SFM names (train split): {sfm_names[n:n+3]}')

            self.colmap_model = ColmapModelReader(self.path_to_colmap_model)

            # self._copy_sfm_images(sfm_names)
            # self._convert_sfm_intrinsics(sfm_names)
            # self._convert_sfm_poses(sfm_names)

            assert path_to_nvm, "Missing input: path_to_nvm"
            self.path_to_nvm = path_to_nvm
            self._generate_from_reconstruction(sfm_names, depth_maps, scene_coordinates)

        else:
            raise(ValueError, 'Specify a source: renders or sfm')


    def _copy_rendered_images(self, names: List[str]):
        """
        Copy images from path_to_renders/images/* to train/rgb/*
        """

        for split in ['train', 'test']:
            path = self.path_to_glace / split / 'rgb'
            if not path.exists(): path.mkdir()

        for i, name in enumerate(names):
            if i < self.num_test: split = 'test'
            else: split = 'train'

            input_image = self.path_to_renders / 'images' / f'{name}.png'
            output_image = self.path_to_glace / split / 'rgb' / f'{name}.png'
            output_image.write_bytes(input_image.read_bytes())
        
        print("Rendered images copied successfully")


    def _copy_sfm_images(self, sfm_names: List[str]):
        """
        Copy images from path_to_colmap_model/../images/* to test/rgb/*
        """

        for split in ['train', 'test']:
            path = self.path_to_glace / split / 'rgb'
            if not path.exists(): path.mkdir()

        path_to_colmap_images = self.path_to_colmap_model.parent / 'images/'

        for i, name in enumerate(sfm_names):
            if i < self.num_test: split = 'test'
            else: split = 'train'

            image = path_to_colmap_images / f'{name}.jpg'
            output_image = self.path_to_glace / split / 'rgb' / f'{name}.jpg'
            output_image.write_bytes(image.read_bytes())

        print("SFM images copied successfully")


    def _convert_rendered_intrinsics(self, names: List[str]):
        """
        Convert path_to_renders/intrinsics/*.txt (w, h, f, f_unit, cx, cy) to train/calibration/*.txt (camera matrix) per image.
        """
        
        for split in ['train', 'test']:
            path = self.path_to_glace / split / 'calibration'
            if not path.exists(): path.mkdir()

        for i, name in enumerate(names):
            if i < self.num_test: split = 'test'
            else: split = 'train'

            file = self.path_to_renders / 'intrinsics' / f'{name}.txt'

            w, h, f, f_unit, cx, cy = file.read_text().strip().split()
            w, h, f, cx, cy = int(w), int(h), float(f), float(cx), float(cy)

            if f_unit == 'MM':
                # convert focal length from mm to pixels
                fx = fy = f * max(w, h) / 36

            else:
                raise NotImplementedError(f"Unit {f_unit} not implemented")

            K = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ])

            output_file = self.path_to_glace / split / 'calibration' / f'{name}.txt'
            np.savetxt(output_file, K, fmt='%15.7e')
        
        print("Rendered intrinsics converted successfully")
    

    def _convert_sfm_intrinsics(self, sfm_names: List[str]):
        """
        Convert SFM intrinsics from colmap_model to train/calibration/*.txt (camera matrix) per image
        """

        for split in ['train', 'test']:
            path = self.path_to_glace / split / 'calibration'
            if not path.exists(): path.mkdir()

        for i, name in enumerate(sfm_names):
            if i < self.num_test: split = 'test'
            else: split = 'train'

            id = self.colmap_model.get_query_image_id(f'{name}.jpg')
            camera_model, w, h, camera_params = self.colmap_model.get_query_intrinsics(id)

            if camera_model == 'PINHOLE':
                fx, fy, cx, cy = camera_params
            else:
                raise NotImplementedError(f"Camera model {camera_model} not implemented")
            
            K = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ])

            output_file = self.path_to_glace / split / 'calibration' / f'{name}.txt'
            np.savetxt(output_file, K, fmt='%15.7e')
        
        print("SFM intrinsics converted successfully")


    def _convert_rendered_poses(self, names: List[str]):
        """
        Convert path_to_renders/poses/*.txt (px, py, pz, qw, qx, qy, qz) to poses/*.txt (transformation matrix) per image
        Format: CAD (inverted Blender camera) -> CAD (conventional camera)
        """

        for split in ['train', 'test']:
            path = self.path_to_glace / split / 'poses'
            if not path.exists(): path.mkdir()

        for i, name in enumerate(names):
            if i < self.num_test: split = 'test'
            else: split = 'train'

            file = self.path_to_renders / 'poses' / f'{name}.txt'

            pose_cad_cam_blender = np.loadtxt(file)
            T_cad_cam_blender = convert_pose_to_matrix(pose_cad_cam_blender)

            T_cad_cam = reverse_camera_pose_for_blender(T_cad_cam_blender,'CAD')

            output_file = self.path_to_glace / split / 'poses' / file.name
            np.savetxt(output_file, T_cad_cam, fmt='%15.7e')
        
        print("Rendered poses converted successfully")


    def _convert_sfm_poses(self, sfm_names: List[str]):
        """
        Convert SFM poses from colmap_model to test/poses/*.txt (transformation matrix) per image
        """

        for split in ['train', 'test']:
            path = self.path_to_glace / split / 'poses'
            if not path.exists(): path.mkdir()

        for i, name in enumerate(sfm_names):
            if i < self.num_test: split = 'test'
            else: split = 'train'

            id = self.colmap_model.get_query_image_id(f'{name}.jpg')
            pose_cam_sfm = self.colmap_model.get_query_pose(id)

            T_cam_sfm = convert_pose_to_matrix(pose_cam_sfm)
            T_sfm_cam = np.linalg.inv(T_cam_sfm)

            T_cad_cam = self.T_cad_sfm @ T_sfm_cam

            output_file = self.path_to_glace / split / 'poses' / f'{name}.txt'
            np.savetxt(output_file, T_cad_cam, fmt='%15.7e')
        
        print("SFM poses converted successfully")
    

    def _convert_rendered_depth_maps(self, names: List[str], to_mm=True):
        """
        Convert depth maps from path_to_renders/depth_maps/*.npz to train/depth_maps/*.npy.
        Change m to mm.
        """

        for split in ['train', 'test']:
            path = self.path_to_glace / split / 'depth'
            if not path.exists(): path.mkdir()
        
        for i, name in enumerate(names):
            if i < self.num_test: split = 'test'
            else: split = 'train'

            depth_map = ModelConversion.convert_depth_map_from_exr_to_numpy(self.path_to_renders / 'depth/', name)
            if to_mm:
                depth_map *= 1000
            output_file = self.path_to_glace / split / 'depth' / f'{name}.npy'
            np.save(output_file, depth_map)
    
        print("Rendered depth maps converted successfully")

    
    def _generate_from_reconstruction(
            self,
            names: List[str],
            depth_maps: bool = True,
            scene_coordinates: bool = True,
            to_mm = True,
            nn_subsampling = 8,
        ):
        """
        Generate GLACE train and test data from NVM file.
        """

        for split in ['train', 'test']:
            path = self.path_to_glace / split / 'depth'
            if not path.exists(): path.mkdir()

            subpaths = ['rgb', 'calibration', 'poses', 'init']
            if depth_maps: subpaths.append('depth')
            if scene_coordinates: subpaths.append('init')

            for subpath in subpaths:
                path = self.path_to_glace / split / subpath
                if not path.exists(): path.mkdir()

        conversion = ModelConversion(T_sfm_cad=self.T_sfm_cad)

        T_sfm_cad = self.T_sfm_cad
        T_cad_sfm = self.T_cad_sfm

        T_sfm_cad = torch.tensor(T_sfm_cad).float()
        T_cad_sfm = torch.tensor(T_cad_sfm).float()

        scale = conversion.s_cad_sfm
        print(f'Scale: {scale}')


        print("Loading SfM reconstruction...")
        file = open(self.path_to_nvm / 'reconstruction.nvm', 'r')
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
            image_file = line[0]
            focal_length = float(line[1])

            name = image_file.split('.')[0]

            if self.num_test > 0 and names.index(name) < self.num_test:
                split = 'test'
            else:
                split = 'train'

            print(f"{cam_idx + 1} / {num_cams}: {image_file} -> {split}")
            print(f"- Focal length: {round(focal_length, 2)}")

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

            pose_file = self.path_to_glace / split / 'poses' / f'{name}.txt'
            np.savetxt(pose_file, T_cad_cam, fmt='%15.7e')


            T_cam_sfm = torch.tensor(T_cam_sfm).float()
            T_sfm_cam = torch.tensor(T_sfm_cam).float()


            # IMAGE
            image = cv.imread(self.path_to_images / image_file)

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
            cv.imwrite(self.path_to_glace / split / 'rgb' / f'{name}.png', image)

            # print(f'Image scale: {img_scale}, Out scale: {out_scale}')


            # INTRINSICS
            intrinsics_file = self.path_to_glace / split / 'calibration' / f'{name}.txt'
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

                    depth_file = self.path_to_glace / split / 'depth' / f'{name}.npy'
                    np.save(depth_file, depth_cad)
                
                if scene_coordinates:

                    valid_mask = coords_sfm.sum(dim=0) != 0

                    coords_sfm_valid = coords_sfm[:, valid_mask]
                    coords_sfm_valid_homogeneous = torch.cat([coords_sfm_valid, torch.ones(1, coords_sfm_valid.size(1))], dim=0)

                    coords_cad_valid_homogeneous = torch.mm(T_cad_sfm, coords_sfm_valid_homogeneous)
                    coords_cad_valid = coords_cad_valid_homogeneous[0:3, :]

                    coords_cad = torch.zeros((3, out_h, out_w))
                    coords_cad[:, valid_mask] = coords_cad_valid

                    coords_file = self.path_to_glace / split / 'init' / f'{name}.dat'
                    torch.save(coords_cad, coords_file)


    def _generate_from_renders(
            self,
            names: List[str],
            depth_maps: bool = True,
            # scene_coordinates: bool,
            to_mm = True,
            nn_subsampling = 1,
        ):
        """
        Generate GLACE train and test data from renders.
        """

        for split in ['train', 'test']:
            path = self.path_to_glace / split / 'depth'
            if not path.exists(): path.mkdir()

            subpaths = ['rgb', 'calibration', 'poses', 'init']
            if depth_maps: subpaths.append('depth')
            # if scene_coordinates: subpaths.append('init')

            for subpath in subpaths:
                path = self.path_to_glace / split / subpath
                if not path.exists(): path.mkdir()
        
        total = len(names)
        for i, name in enumerate(names):
            if i < self.num_test: split = 'test'
            else: split = 'train'

            print(f"{i + 1} / {total}: {name} -> {split}")

            # POSE
            pose_cad_cam_blender = np.loadtxt(self.path_to_renders / 'poses' / f'{name}.txt')
            T_cad_cam_blender = convert_pose_to_matrix(pose_cad_cam_blender)

            T_cad_cam = reverse_camera_pose_for_blender(T_cad_cam_blender,'CAD')

            pose_file = self.path_to_glace / split / 'poses' / f'{name}.txt'
            np.savetxt(pose_file, T_cad_cam, fmt='%15.7e')


            # IMAGE
            image = cv.imread(self.path_to_renders / 'images' / f'{name}.png')

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
            cv.imwrite(self.path_to_glace / split / 'rgb' / f'{name}.png', image)

            # print(f'Image scale: {img_scale}, Out scale: {out_scale}')


            # INTRINSICS
            file = self.path_to_renders / 'intrinsics' / f'{name}.txt'

            w, h, f, f_unit, _, _ = file.read_text().strip().split()
            w, h, f = int(w), int(h), float(f)

            if f_unit == 'MM':
                # convert focal length from mm to pixels
                focal_length = f * max(w, h) / 36
                focal_length *= img_scale
            else:
                raise NotImplementedError(f"Unit {f_unit} not implemented")
            
            intrinsics_file = self.path_to_glace / split / 'calibration' / f'{name}.txt'
            with open(intrinsics_file, 'w') as f:
                f.write(str(focal_length * img_scale))

            # DEPTH MAPS & SCENE COORDINATES
            # if depth_maps or scene_coordinates:
                
            if depth_maps:
                    
                depth_map = ModelConversion.convert_depth_map_from_exr_to_numpy(self.path_to_renders / 'depth/', name)

                # resize depth map & subsample by using nearest neighbor
                depth_map_subsampled = np.zeros((out_h, out_w))

                # print(f"Resizing from {depth_map.shape} to {depth_map_subsampled.shape}")
                scale_y = out_h / depth_map.shape[0]
                scale_x = out_w / depth_map.shape[1]

                for y in range(out_h):
                    for x in range(out_w):
                        x_ = int( (x+0.5) / scale_x - 0.5)
                        y_ = int( (y+0.5) / scale_y - 0.5)

                        depth_map_subsampled[y, x] = depth_map[y_, x_]

                #         if x==0 and y==0: print(f'({x}, {y}) <- ({x_}, {y_})')
                # print(f'({x}, {y}) <- ({x_}, {y_})')
                

                depth_cad = depth_map_subsampled
                assert depth_cad.shape == (out_h, out_w), depth_cad.shape

                if to_mm:
                    depth_map_subsampled *= 1000

                depth_file = self.path_to_glace / split / 'depth' / f'{name}.npy'
                np.save(depth_file, depth_map_subsampled)

            # SCENE COORDINATES
            # if scene_coordinates:

            #     x, y = np.meshgrid(np.arange(out_w), np.arange(out_h))
            #     x = (x - out_w / 2) * depth_cad / (focal_length * out_scale)
            #     y = (y - out_h / 2) * depth_cad / (focal_length * out_scale)
            #     z = depth_cad

            #     ones = np.ones_like(z)
            #     ones = np.where(z == 0, 0, ones)

            #     coords_cam = np.stack([x, y, z, ones], axis=-1)
            #     coords_cam = coords_cam.reshape(-1, 4)
                
            #     coords_cad = coords_cam @ T_cad_cam.T
            #     coords_cad = coords_cad[:, :3]
            #     coords_cad = coords_cad.reshape(depth_cad.shape[0], depth_cad.shape[1], 3)

            #     assert coords_cad.shape == (out_h, out_w, 3), coords_cad.shape

            #     coords_cad = torch.tensor(coords_cad).float().permute(2, 0, 1)

            #     assert coords_cad.size() == (3, out_h, out_w), coords_cad.size()

            #     coords_file = self.path_to_glace / split / 'init' / f'{name}.dat'
            #     torch.save(coords_cad, coords_file)


if __name__ == '__main__':

    T_notre_dame = np.array([
        [-0.04308, -0.07366, -0.0008805, -1.525],
        [0.0245, -0.01336, -0.08065, 4.145],
        [0.06947, -0.04097, 0.02789, 10.74],
        [0, 0, 0, 1],
    ])

    T_st_peters = np.array([
        [-0.008938, 0.04505, -4.739e-05, 7.153],
        [-0.01353, -0.002731, -0.04381, 1.885],
        [-0.04297, -0.008511, 0.01381, 5.6],
        [0, 0, 0, 1],
    ])

    T_pantheon = np.array([
        [-0.1956, 0.005829, -0.0004737, -0.05305],
        [-0.001383, -0.06151, -0.1857, 5.694],
        [-0.005682, -0.1856, 0.06152, 13.51],
        [0, 0, 0, 1],
    ])

    from data import path_to_data


    # Pantheon

    path_to_colmap_model = path_to_data / '3D Models/Pantheon/Reference/dense/sparse'

    GlaceConversion(
        source='SFM',
        path_to_colmap_model=path_to_colmap_model,
        path_to_glace=path_to_data / 'GLACE/pantheon (SFM)',
        path_to_nvm=path_to_colmap_model,
        T_sfm_cad=T_pantheon,
        num_test=100,
        depth_maps=True,
        scene_coordinates=True,
    )

    GlaceConversion(
        source='renders',
        path_to_glace=path_to_data / 'GLACE/pantheon (renders)',
        path_to_renders=path_to_data / 'Evaluation/pantheon B/ground truth/renders',
        num_test=100,
        depth_maps=True,
        scene_coordinates=True,
    )


    # Notre Dame B

    # GlaceConversion(
    #     source='renders',
    #     path_to_glace=path_to_data / 'GLACE/notre dame B (orbit renders)/',
    #     path_to_renders=path_to_data / 'Evaluation/notre dame B/inputs/database/',
    # )

    # GlaceConversion(
    #     source='renders',
    #     path_to_glace=path_to_data / 'GLACE/notre dame B (SFM renders)/',
    #     path_to_renders=path_to_data / 'Evaluation/notre dame B (SFM)/ground truth/renders/',
    #     num_test=100,
    # )

    # path_to_colmap_model = path_to_data / '3D Models/Notre Dame/Reference/dense/sparse'
    # # - SFM
    # GlaceConversion(
    #     source='SFM',
    #     path_to_colmap_model= path_to_colmap_model,
    #     path_to_glace=Path('/Users/eric/Documents/Studies/MSc Robotics/Thesis/Data/GLACE/notre dame (SFM)'),
    #     T_sfm_cad=T_notre_dame,
    #     num_test=0,
    #     depth_maps=True,
    #     path_to_nvm=path_to_colmap_model,
    # )


    # St Peters Square B

    # - Renders
    # GlaceConversion(
    #    source='renders',
    #     path_to_renders=Path('/home/johndoe/Documents/data/Evaluation/st peters square B/ground truth/renders/'),
    #     path_to_glace=Path('/home/johndoe/Documents/data/GLACE/st peters square B (SFM renders)/'),
    # )

    # - SFM
    # GlaceConversion(
    #     source='SFM',
    #     path_to_colmap_model=Path('/home/johndoe/Documents/data/3D Models/St Peters Square/Reference/dense/sparse/'),
    #     path_to_glace=Path('/home/johndoe/Documents/data/GLACE/st peters square B (SFM renders)/'),
    #     T_sfm_cad=T_st_peters,
    #     num_test=100,
    # )
