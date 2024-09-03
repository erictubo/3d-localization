from pathlib import Path
from typing import List
import numpy as np
from transformations import decompose_matrix, compose_matrix

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
            sfm_names: list = None, # list of sfm names, else all images from SFM
            T_sfm_cad: np.ndarray = None,
            num_test: int = 0,
        ):

        self.path_to_glace = path_to_glace
        for split in ['train', 'test']:
            path = self.path_to_glace / split
            if not path.exists(): path.mkdir()

        self.num_test = num_test

        if source.lower() == 'renders':
            assert path_to_renders, "Missing input: path_to_renders"
            self.path_to_renders = path_to_renders

            names = [image.name for image in self.path_to_renders.glob('images/*')]
            names = [name.split('.')[0] for name in names]
            names.sort()

            self.copy_rendered_images(names)
            self.convert_rendered_intrinsics(names)
            self.convert_rendered_poses(names)
            self.convert_rendered_depth_maps(names)
        
        elif source.lower() == 'sfm':
            assert path_to_colmap_model, "Missing input: path_to_colmap_model"
            self.path_to_colmap_model = path_to_colmap_model

            assert T_sfm_cad is not None, "Missing input: reference transformation T_sfm_cad"
            assert T_sfm_cad.shape == (4,4), "Reference transformation is not a 4x4 matrix"
            self.T_sfm_cad = T_sfm_cad
            self.T_cad_sfm = np.linalg.inv(self.T_sfm_cad)

            if not sfm_names:
                sfm_names = [image.name for image in self.path_to_colmap_model.parent.glob('images/*')]
            sfm_names = [name.split('.')[0] for name in sfm_names]
            sfm_names.sort()

            n = self.num_test
            if n != 0: print('SFM names (test split):', sfm_names[:3])
            print('SFM names (train split):', sfm_names[n:n+3])

            self.colmap_model = ColmapModelReader(self.path_to_colmap_model)

            self.copy_sfm_images(sfm_names)
            self.convert_sfm_intrinsics(sfm_names)
            self.convert_sfm_poses(sfm_names)

        else:
            raise(ValueError, 'Specify a source: renders or sfm')
        
        # TODO: implement scene coordinates
        # Make sure they are in the correct frame as well (CAD)
        # self.copy_scene_coordinates()

        # INFO: for SFM reconstruction -> scene coordinates
        # see GLACE setup_cambridge.py

    def convert_rendered_depth_maps(self, names: List[str]):
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
            depth_map_mm = depth_map * 1000
            output_file = self.path_to_glace / split / 'depth' / f'{name}.npy'
            np.save(output_file, depth_map_mm)


    # def convert_scene_coordinates(self):
    #     """
    #     Convert scene coordinates from 
    #     """

    #     # INPUT: path_to_renders/scene_coordinates/*.npz
    #     # npz: dict['scene_coordinates'] = scene_coordinates ndarray (w x h x 3)
    #     # w, h = image width, height

    #     # OUTPUT: train/init/* torch tensors (3 x H x W)
    #     # H, W = SCR output dimension = 640 x 480 (480 fixed, width based on aspect ratio)
    #     # invalid set to zero

    #     pass


    def copy_rendered_images(self, names: List[str]):
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
        
        print("Database images copied successfully")


    def convert_rendered_intrinsics(self, names: List[str]):
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
        
        print("Database intrinsics converted successfully")


    def convert_rendered_poses(self, names: List[str]):
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
        
        print("Database poses converted successfully")


    def copy_sfm_images(self, sfm_names: List[str]):
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
    

    def convert_sfm_intrinsics(self, sfm_names: List[str]):
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


    def convert_sfm_poses(self, sfm_names: List[str]):
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
    

    def copy_scene_coordinates():
        pass


if __name__ == '__main__':

    T_notre_dame = np.array([
        [-0.04308, -0.07366, -0.0008805, -1.525],
        [0.0245, -0.01336, -0.08065, 4.145],
        [0.06947, -0.04097, 0.02789, 10.74],
        [0, 0, 0, 1]
    ])

    T_st_peters = np.array([
        [-0.008938, 0.04505, -4.739e-05, 7.153],
        [-0.01353, -0.002731, -0.04381, 1.885],
        [-0.04297, -0.008511, 0.01381, 5.6],
        [0, 0, 0, 1]
    ])

    # path_to_data = Path('/Users/eric/Documents/Studies/MSc Robotics/Thesis/Data/')
    path_to_data = Path('/home/johndoe/Documents/data/')



    # GlaceConversion(
    #     source='renders',
    #     path_to_glace=path_to_data / 'GLACE/notre dame B (orbit renders)/',
    #     path_to_renders=path_to_data / 'Evaluation/notre dame B/inputs/database/',
    # )

    GlaceConversion(
        source='renders',
        path_to_glace=path_to_data / 'GLACE/notre dame B (SFM renders)/',
        path_to_renders=path_to_data / 'Evaluation/notre dame B (SFM)/ground truth/renders/',
        num_test=100,
    )


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
