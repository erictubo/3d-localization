from pathlib import Path
from typing import List
import numpy as np
from transformations import decompose_matrix, compose_matrix

from colmap_model import ColmapModelReader
from model_conversion import *


class GlaceConversion:
    """
    Conversion of data for compatibility with GLACE
    - Read from input/database/* and input/query/* (using COLMAP data)
    - Write to train/*, test/*
    """

    def __init__(
            self,
            path_to_input: Path,
            path_to_output: Path,
            renders: bool,
            sfm_data: bool,
            sfm_names: list = None, # list of sfm names, else uses images in input/{renders_prefix}/images/*
            path_to_colmap_model: Path = None,
            T_sfm_cad: np.ndarray = None,
            renders_prefix: str = 'database/',
            sfm_prefix: str = 'query/',
            train_prefix: str = 'train/',
            test_prefix: str = 'test/',
        ):

        self.path_to_input = path_to_input
        self.path_to_input_renders = path_to_input / renders_prefix

        self.path_to_output = path_to_output
        self.path_to_output_train = path_to_output / train_prefix
        self.path_to_output_test = path_to_output / test_prefix
        
        if not self.path_to_output_train.exists(): self.path_to_output_train.mkdir()
        if not self.path_to_output_test.exists(): self.path_to_output_test.mkdir()

        if renders:
            self.copy_rendered_images()
            self.convert_rendered_intrinsics()
            self.convert_rendered_poses()
        
        if sfm_data:
            self.path_to_input_sfm = path_to_input / sfm_prefix
            assert path_to_colmap_model, "path_to_colmap_model must be provided for sfm data conversion"
            self.path_to_colmap_model = path_to_colmap_model

            assert T_sfm_cad.shape == (4,4), "Reference transformation is not a 4x4 matrix"
            self.T_sfm_cad = T_sfm_cad
            self.T_cad_sfm = np.linalg.inv(self.T_sfm_cad)
            
            self.colmap_model = ColmapModelReader(self.path_to_colmap_model)

            if not sfm_names:
                sfm_names = [image.name for image in self.path_to_input_query.glob('images/*')]
            sfm_names = [name.split('.')[0] for name in sfm_names]

            self.copy_sfm_images(sfm_names)
            self.convert_sfm_intrinsics(sfm_names)
            self.convert_sfm_poses(sfm_names)

        # TODO: test new method for (1) renders and (2) sfm_data
        
        # TODO: implement scene coordinates
        # Make sure they are in the correct frame as well (CAD)
        # self.copy_scene_coordinates()


    def copy_rendered_images(self):
        """
        Copy images from input/database/images/* to train/rgb/*
        """

        database_images = list(self.path_to_input_renders.glob('images/*'))
        self.num_database_images = len(database_images)
        print(f"Number of database images: {self.num_database_images}")

        path_to_output_rgb = self.path_to_output_train / 'rgb/'
        if not path_to_output_rgb.exists(): path_to_output_rgb.mkdir()

        for input_image in database_images:
            output_image = path_to_output_rgb / input_image.name
            output_image.write_bytes(input_image.read_bytes())
        
        print("Database images copied successfully")


    def convert_rendered_intrinsics(self):
        """
        Convert database/intrinsics/*.txt (w, h, f, f_unit, cx, cy) to train/calibration/*.txt (camera matrix) per image.
        """
        
        database_intrinsics = list(self.path_to_input_renders.glob('intrinsics/*.txt'))
        self.num_database_intrinsics = len(database_intrinsics)

        assert self.num_database_intrinsics == self.num_database_images, \
            f"Number of intrinsics files ({self.num_database_intrinsics}) does not match number of images ({self.num_database_images})"

        path_to_output_calibration = self.path_to_output_train / 'calibration/'
        if not path_to_output_calibration.exists(): path_to_output_calibration.mkdir()

        for file in database_intrinsics:
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

            output_file = path_to_output_calibration / file.name
            np.savetxt(output_file, K, fmt='%15.7e')
        
        print("Database intrinsics converted successfully")


    def convert_rendered_poses(self):
        """
        Convert database/poses/*.txt (px, py, pz, qw, qx, qy, qz) to poses/*.txt (transformation matrix) per image
        Format: CAD (inverted Blender camera) -> CAD (conventional camera)
        """

        database_poses = list(self.path_to_input_renders.glob('poses/*.txt'))
        self.num_database_poses = len(database_poses)

        assert self.num_database_poses == self.num_database_images, \
            f"Number of poses files ({self.num_database_poses}) does not match number of images ({self.num_database_images})"

        path_to_output_poses = self.path_to_output_train / 'poses/'
        if not path_to_output_poses.exists(): path_to_output_poses.mkdir()

        for file in database_poses:
            pose_cad_cam_blender = np.loadtxt(file)
            T_cad_cam_blender = convert_pose_to_matrix(pose_cad_cam_blender)

            T_cad_cam = reverse_camera_pose_for_blender(T_cad_cam_blender, frame='CAD')

            output_file = path_to_output_poses / file.name
            np.savetxt(output_file, T_cad_cam, fmt='%15.7e')
        
        print("Database poses converted successfully")


    def copy_sfm_images(self, sfm_names: List[str]):
        """
        Copy images from path_to_colmap_model/../images/* to test/rgb/*
        """

        path_to_output_rgb = self.path_to_output_test / 'rgb/'
        if not path_to_output_rgb.exists(): path_to_output_rgb.mkdir()

        path_to_colmap_images = self.path_to_colmap_model.parent / 'images/'
        for name in sfm_names:
            image = path_to_colmap_images / f'{name}.jpg'
            output_image = path_to_output_rgb / f'{name}.jpg'
            output_image.write_bytes(image.read_bytes())

        print("Query images copied successfully")
        
    

    def convert_sfm_intrinsics(self, sfm_names: List[str]):
        """
        Convert query intrinsics from colmap_model to train/calibration/*.txt (camera matrix) per image
        """

        path_to_output_calibration = self.path_to_output_test / 'calibration/'
        if not path_to_output_calibration.exists(): path_to_output_calibration.mkdir()

        for name in sfm_names:
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

            output_file = path_to_output_calibration / f'{name}.txt'
            np.savetxt(output_file, K, fmt='%15.7e')
        
        print("Query intrinsics converted successfully")


    def convert_sfm_poses(self, sfm_names: List[str]):
        """
        Convert query poses from colmap_model to test/poses/*.txt (transformation matrix) per image
        """

        path_to_output_poses = self.path_to_output_test / 'poses/'
        if not path_to_output_poses.exists(): path_to_output_poses.mkdir()

        for name in sfm_names:
            id = self.colmap_model.get_query_image_id(f'{name}.jpg')
            pose_cam_sfm = self.colmap_model.get_query_pose(id)

            T_cam_sfm = convert_pose_to_matrix(pose_cam_sfm)
            T_sfm_cam = np.linalg.inv(T_cam_sfm)

            T_cad_cam = self.T_cad_sfm @ T_sfm_cam

            output_file = path_to_output_poses / f'{name}.txt'
            np.savetxt(output_file, T_cad_cam, fmt='%15.7e')
        
        print("Query poses converted successfully")
    

    def copy_scene_coordinates():
        pass


# class FrameConversion:

#     def __init__(
#             self,
#             T_sfm_cad: float,
#             path_to_input: Path,
#             path_to_output: Path,
#             type: str = 'matrix',
#         ):
#         self.T_sfm_cad = T_sfm_cad
#         self.T_cad_sfm = np.linalg.inv(T_sfm_cad)

#         self.scale, shear, angles, translate, perspective = decompose_matrix(self.T_cad_sfm)
#         print('Scale: ', self.scale)

#         self.path_to_input = path_to_input
#         self.path_to_output = path_to_output
        
#         if not self.path_to_input.exists():
#             raise FileNotFoundError(f"{self.path_to_input} does not exist")
#         if not self.path_to_output.exists(): self.path_to_output.mkdir()

#         if type != 'matrix':
#             raise NotImplementedError(f"Type {type} not implemented")
        
#     def reverse_camera_direction(self):
#         for file in self.path_to_input.glob('*.txt'):
#             T = np.loadtxt(file)
#             assert T.shape == (4, 4), f"Pose matrix shape {T.shape} is not 4x4"
#             T_rev = reverse_camera_pose_for_blender(T, frame='CAD')
#             output_file = self.path_to_output / file.name
#             np.savetxt(output_file, T_rev, fmt='%15.7e')

#     def convert_cad_to_sfm_frame(self, reverse_camera):
#         """
#         Convert pose matrices using conversion matrix.
#         """
        
#         for file in self.path_to_input.glob('*.txt'):
#             T = np.loadtxt(file)
#             assert T.shape == (4, 4), f"Pose matrix shape {T.shape} is not 4x4"

#             if reverse_camera:
#                 T = reverse_camera_pose_for_blender(T, frame='SFM')
#             T = self.T_sfm_cad @ T

#             output_file = self.path_to_output / file.name
#             np.savetxt(output_file, T, fmt='%15.7e')


#     def scale_poses(self, up=True):
#         """
#         Scale pose matrices by self.scale and save to path
#         """

#         for file in self.path_to_input.glob('*.txt'):
#             T = np.loadtxt(file)
#             assert T.shape == (4, 4), f"Pose matrix shape {T.shape} is not 4x4"
#             if up:
#                 T[:3, 3] *= self.scale
#             else:
#                 T[:3, 3] /= self.scale

#             output_file = self.path_to_output / file.name
#             np.savetxt(output_file, T, fmt='%15.7e')



if __name__ == '__main__':

    T_notre_dame = np.array([
        [-0.04308, -0.07366, -0.0008805, -1.525],
        [0.0245, -0.01336, -0.08065, 4.145],
        [0.06947, -0.04097, 0.02789, 10.74],
        [0, 0, 0, 1]
    ])

    path_to_data = Path('/home/johndoe/Documents/data/GLACE/')

    # for name in ['notre dame B (real)', 'notre dame B (SFM renders)']:
    #     for split in ['train', 'test']:
    
    #         FrameConversion(
    #             T_sfm_cad=T_notre_dame,
    #             path_to_input=path_to_data / name / split / 'poses CAD Blender',
    #             path_to_output=path_to_data / name / split / 'poses SFM rev',
    #         ).convert_cad_to_sfm_frame(reverse_camera=True)




    # GlaceConversion(
    #     path_to_input=Path('/Users/eric/Documents/Studies/MSc Robotics/Thesis/Evaluation/notre_dame_B_new/inputs/'),
    #     path_to_output=Path('/Users/eric/Developer/glace/datasets/notre dame B/'),
    #     renders=True,
    #     query=True,
    #     sfm_names=[
    #             '00145492_2294287255.jpg',
    #             '00870470_3859452456.jpg',
    #             '01133173_279144982.jpg',
    #             '01333333_3920521666.jpg',
    #             '05270949_6179349375.jpg',
    #             '05353713_5401929533.jpg',
    #             '07985698_4822004965.jpg',
    #         ],
    #     path_to_colmap_model=Path('/Users/eric/Documents/Studies/MSc Robotics/Thesis/3D Models/Notre Dame/Reference/dense/sparse/')
    # )


    # Notre Dame B: SFM poses
    GlaceConversion(
        path_to_input=Path('/home/johndoe/Documents/data/Evaluation/st peters square B/ground truth/'),
        path_to_output=Path('/home/johndoe/Documents/data/GLACE/st peters square B (SFM renders)/'),
        renders=True,
        renders_prefix='renders/',
        sfm_data=False,
    )


    # # Notre Dame B with orbit renders
    # GlaceConversion(
    #     path_to_input=Path('/home/johndoe/Documents/data/Evaluation/notre dame B/inputs/'),
    #     path_to_output=Path('/home/johndoe/Documents/data/GLACE/notre dame B (orbit renders)/'),
    #     renders=True,
    #     sfm_data=False,
    #     # sfm_names=None,
    #     # path_to_colmap_model=Path('/home/johndoe/Documents/data/3D Models/Notre Dame/Reference/dense/sparse')
    # )
