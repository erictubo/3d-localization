from pathlib import Path
from typing import List
import numpy as np

from colmap_model import ColmapModelReader
from model_conversion import convert_pose_to_matrix


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
            database: bool,
            query: bool,
            query_names: list = None, # list of query, else uses images in input/query/images/*
            path_to_colmap_model: Path = None,
            database_prefix: str = 'database/',
            query_prefix: str = 'query/',
            train_prefix: str = 'train/',
            test_prefix: str = 'test/',
        ):

        self.path_to_input = path_to_input
        self.path_to_input_database = path_to_input / database_prefix
        if query:
            self.path_to_input_query = path_to_input / query_prefix

        self.path_to_output = path_to_output
        self.path_to_output_train = path_to_output / train_prefix
        self.path_to_output_test = path_to_output / test_prefix
        
        if not self.path_to_output_train.exists(): self.path_to_output_train.mkdir()
        if not self.path_to_output_test.exists(): self.path_to_output_test.mkdir()

        if database:
            self.copy_database_images()
            self.convert_database_intrinsics()
            self.convert_database_poses()
        
        if query:
            assert path_to_colmap_model, "path_to_colmap_model must be provided for query conversion"
            self.path_to_colmap_model = path_to_colmap_model
            self.colmap_model = ColmapModelReader(self.path_to_colmap_model)

            if not query_names:
                query_names = [image.name for image in self.path_to_input_query.glob('images/*')]
            query_names = [name.split('.')[0] for name in query_names]

            self.copy_query_images(query_names)
            self.convert_query_intrinsics(query_names)
            self.convert_query_poses(query_names)
        
        # TODO: implement scene coordinates
        # self.copy_scene_coordinates()


    def copy_database_images(self):
        """
        Copy images from input/database/images/* to train/rgb/*
        """

        database_images = list(self.path_to_input_database.glob('images/*'))
        self.num_database_images = len(database_images)
        print(f"Number of database images: {self.num_database_images}")

        path_to_output_rgb = self.path_to_output_train / 'rgb/'
        if not path_to_output_rgb.exists(): path_to_output_rgb.mkdir()

        for input_image in database_images:
            output_image = path_to_output_rgb / input_image.name
            output_image.write_bytes(input_image.read_bytes())
        
        print("Database images copied successfully")


    def convert_database_intrinsics(self):
        """
        Convert database/intrinsics/*.txt (w, h, f, f_unit, cx, cy) to train/calibration/*.txt (camera matrix) per image.
        """
        
        database_intrinsics = list(self.path_to_input_database.glob('intrinsics/*.txt'))
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


    def convert_database_poses(self):
        """
        Convert database/poses/*.txt (px, py, pz, qw, qx, qy, qz) to poses/*.txt (transformation matrix) per image
        """

        database_poses = list(self.path_to_input_database.glob('poses/*.txt'))
        self.num_database_poses = len(database_poses)

        assert self.num_database_poses == self.num_database_images, \
            f"Number of poses files ({self.num_database_poses}) does not match number of images ({self.num_database_images})"

        path_to_output_poses = self.path_to_output_train / 'poses/'
        if not path_to_output_poses.exists(): path_to_output_poses.mkdir()

        for file in database_poses:
            pose = np.loadtxt(file)
            pose_matrix = convert_pose_to_matrix(pose)

            output_file = path_to_output_poses / file.name
            np.savetxt(output_file, pose_matrix, fmt='%15.7e')
        
        print("Database poses converted successfully")


    def copy_query_images(self, query_names: List[str]):
        """
        Copy images from path_to_colmap_model/../images/* to test/rgb/*
        """

        path_to_output_rgb = self.path_to_output_test / 'rgb/'
        if not path_to_output_rgb.exists(): path_to_output_rgb.mkdir()

        path_to_colmap_images = self.path_to_colmap_model.parent / 'images/'
        for name in query_names:
            image = path_to_colmap_images / f'{name}.jpg'
            output_image = path_to_output_rgb / f'{name}.jpg'
            output_image.write_bytes(image.read_bytes())

        print("Query images copied successfully")
        
    

    def convert_query_intrinsics(self, query_names: List[str]):
        """
        Convert query intrinsics from colmap_model to train/calibration/*.txt (camera matrix) per image
        """

        path_to_output_calibration = self.path_to_output_test / 'calibration/'
        if not path_to_output_calibration.exists(): path_to_output_calibration.mkdir()

        for name in query_names:
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


    def convert_query_poses(self, query_names: List[str]):
        """
        Convert query poses from colmap_model to test/poses/*.txt (transformation matrix) per image
        """

        path_to_output_poses = self.path_to_output_test / 'poses/'
        if not path_to_output_poses.exists(): path_to_output_poses.mkdir()

        for name in query_names:
            id = self.colmap_model.get_query_image_id(f'{name}.jpg')
            pose = self.colmap_model.get_query_pose(id)

            pose_matrix = convert_pose_to_matrix(pose)

            output_file = path_to_output_poses / f'{name}.txt'
            np.savetxt(output_file, pose_matrix, fmt='%15.7e')
        
        print("Query poses converted successfully")
    

    def copy_scene_coordinates():
        pass


class ScaleConversion:

    def __init__(
            self,
            scale: float,
            path_to_input: Path,
            path_to_output: Path,
            type: str = 'matrix',
        ):
        self.scale = scale
        self.path_to_input = path_to_input
        self.path_to_output = path_to_output
        
        if not self.path_to_input.exists():
            raise FileNotFoundError(f"{self.path_to_input} does not exist")
        if not self.path_to_output.exists(): self.path_to_output.mkdir()

        if type != 'matrix':
            raise NotImplementedError(f"Type {type} not implemented")

        self.scale_poses()

    def scale_poses(self):
        """
        Scale pose matrices by self.scale and save to path
        """

        for file in self.path_to_input.glob('*.txt'):
            pose = np.loadtxt(file)
            assert pose.shape == (4, 4), f"Pose matrix shape {pose.shape} is not 4x4"
            pose[:3, 3] *= self.scale

            output_file = self.path_to_output / file.name
            np.savetxt(output_file, pose, fmt='%15.7e')


if __name__ == '__main__':

    scale_notre_dame = 11.718009811983837
    scale_reichstag  = 15.304415226012821
    scale_st_peters  = 21.77207755406891
    

    ScaleConversion(
        scale=11.718,
        path_to_input=Path('/Users/eric/Developer/glace/datasets/notre_dame_B/train/poses/'),
        path_to_output=Path('/Users/eric/Developer/glace/datasets/notre_dame_B/train/scaled_poses/'),
    )




    # GlaceConversion(
    #     path_to_input=Path('/Users/eric/Documents/Studies/MSc Robotics/Thesis/Evaluation/notre_dame_B_new/inputs/'),
    #     path_to_output=Path('/Users/eric/Developer/glace/datasets/notre dame B/'),
    #     database=True,
    #     query=True,
    #     query_names=[
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


    # Notre Dame B with orbit renders
    # GlaceConversion(
    #     path_to_input=Path('/home/johndoe/Documents/data/Evaluation/notre dame B/inputs/'),
    #     path_to_output=Path('/home/johndoe/Documents/data/GLACE/notre dame B rendered/'),
    #     database=True,
    #     query=True,
    #     query_names=None,
    #     path_to_colmap_model=Path('/home/johndoe/Documents/data/3D Models/Notre Dame/Reference/dense/sparse')
    # )

    # Notre Dame B with real images as training


    # Notre Dame B with rendered SFM poses
    # GlaceConversion(
    #     path_to_input=Path('/home/johndoe/Documents/data/Evaluation/notre dame B/ground truth/'),
    #     path_to_output=Path('/home/johndoe/Documents/data/GLACE/notre dame B rendered/'),
    #     database_prefix='renders/',
    #     database=True,
    #     query=False,
    # )