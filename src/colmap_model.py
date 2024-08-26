import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List
from math import pi

from colmap.read_write_model import *


class ColmapModelReader:
    """
    Import SFM model data from COLMAP database (ground truth):
    - get image id -> pose, intrinsics
    - write intrinsics & poses to text file (as queries for MeshLoc pipeline)
    """

    def __init__(self, path_to_data: Path):

        self.path_to_data = path_to_data
        self.cameras, self.images, self.points3d = read_model(path_to_data)
        """
        - images[image_id] = Image(id, qvec, tvec, camera_id, name, xys, point3D_ids)   
        - cameras[camera_id] = Camera(id, model, width, height, params)
        - points3D[point3D_id] = Point3D(id, xyz, rgb, error, image_ids, point2D_idxs)
        """
    
    def get_all_image_names(self) -> List[str]:
        """
        Get all image names.
        """
        image_names = [image.name for image in self.images.values()]
        image_names.sort()
        
        return image_names
        
    def get_query_image_id(self, image_name: str) -> int:
        """
        Get image ID from image name.
        """
        image_name = image_name.split('/')[-1]
        image_id = [k for k, v in self.images.items() if v.name == image_name][0]
        return image_id
    
    def get_query_pose(self, image_id: int) -> np.ndarray:
        """
        Get pose of image from database.
        """
        image = self.images[image_id]
        pose = np.concatenate([image.tvec, image.qvec])
        assert pose.shape == (7,), pose.shape
        return pose
    
    def get_query_intrinsics(self, image_id: int) -> Tuple[str, int, int, np.ndarray]:
        """
        Get intrinsics of image from database.
        Output: camera model, camera parameters, image width, image height
        """
        image = self.images[image_id]
        camera_id = image.camera_id

        camera = self.cameras[camera_id]

        camera_model: str = camera.model
        w: int = camera.width
        h: int = camera.height
        camera_params: np.ndarray = camera.params
        
        return camera_model, w, h, camera_params

    def write_query_intrinsics_text_file(
            self,
            path_to_output: Path,
            query_names: List[str],
            file_name: str = 'queries.txt',
        ):
        """
        Write queries text file in inputs directory (for use with MeshLoc).
        Format in each line: query_name, camera_model, w, h, camera_params
        """
        if not file_name.endswith('.txt'):
            file_name += '.txt'
        with open(path_to_output / file_name, 'w') as f:
            for query_name in query_names:
                query_id = self.get_query_image_id(query_name)
                (camera_model, w, h, camera_params) = self.get_query_intrinsics(query_id)
                f.write(f"{query_name} {camera_model} {w} {h} {' '.join(map(str, camera_params))}\n")

    def write_query_poses_text_file(
            self,
            path_to_output: Path,
            query_names: List[str],
            file_name: str = 'cam_sfm_poses.txt',
        ):
        """
        Write poses ground truth text file in outputs directory (for use with MeshLoc).
        Format in each line: query_name, pose
        """
        if not file_name.endswith('.txt'):
            file_name += '.txt'
        with open(path_to_output / file_name, 'w') as f:
            for query_name in query_names:
                query_id = self.get_query_image_id(query_name)
                pose = self.get_query_pose(query_id)
                f.write(f"{query_name} {' '.join(map(str, pose))}\n")

    
    @staticmethod
    def write_poses_text_file(
            poses: Dict[str, np.ndarray],
            path_to_output: Path,
            file_name: str,
            quaternion_first: bool, # for writing
        ):
        """
        Write poses text file.
        Assumes input with translation first.
        """
        with open(path_to_output / file_name, 'w') as file:
            for query_name, pose in poses.items():
                tvec, qvec = pose[:3], pose[3:]
                if quaternion_first:
                    file.write(f"{query_name} {' '.join(map(str, qvec))} {' '.join(map(str, tvec))}\n")
                else:
                    file.write(f"{query_name} {' '.join(map(str, tvec))} {' '.join(map(str, qvec))}\n")
    
    @staticmethod
    def write_intrinsics_text_file(
            intrinsics: Dict[str, Tuple[str, int, int, np.ndarray]],
            path_to_output: Path,
            file_name: str,
        ):
        """
        Write intrinsics text file.
        """
        with open(path_to_output / file_name, 'w') as file:
            for query_name, (camera_model, w, h, camera_params) in intrinsics.items():
                file.write(f"{query_name} {camera_model} {w} {h} {' '.join(map(str, camera_params))}\n")


class ColmapModelWriter:

    """
    - images[image_id] = Image(id, qvec, tvec, camera_id, name, xys, point3D_ids)   
    - cameras[camera_id] = Camera(id, model, width, height, params)
    """

    @staticmethod
    def read_intrinsics_text_file(
            path_to_intrinsics: Path,
            intrinsics_file: str,
        ) -> Dict[str, Tuple[str, int, int, np.ndarray]]:
        """
        Read intrinsics text file.
        """
        intrinsics = {}

        with open(path_to_intrinsics / intrinsics_file, 'r') as file:
            intrinsics_data = file.readlines()
            for line in intrinsics_data:
                items = line.strip().split(' ')
                query_name = items[0]
                camera_model = items[1]
                w, h = int(items[2]), int(items[3])
                camera_params = np.array([float(i) for i in items[4:]])
                intrinsics[query_name] = (camera_model, w, h, camera_params)

        return intrinsics

    @staticmethod
    def read_poses_text_file(
            path_to_poses: Path,
            poses_file: str,
            quaternion_first: bool,
        ) -> Dict[str, np.ndarray]:
        """
        Read poses text file.
        """
        poses = {}

        with open(path_to_poses / poses_file, 'r') as file:
            poses_data = file.readlines()
            for line in poses_data:
                items = line.strip().split(' ')
                query_name = items[0]
                pose = np.array([float(i) for i in items[1:]])
                if quaternion_first:
                    qvec, tvec = pose[:4], pose[4:]
                else:
                    tvec, qvec = pose[:3], pose[3:]
                poses[query_name] = np.concatenate([tvec, qvec])

        return poses
    
    @staticmethod
    def write_poses_dict_to_colmap_format(
            output_path: Path,
            poses: Dict[str, np.ndarray],
        ):
        """
        Write poses to COLMAP format.
        """
        images = {}
        image_id = 0

        for query_name, pose in poses.items():
            image_id += 1
            camera_id = image_id
            tvec, qvec = pose[:3], pose[3:]
            images[image_id] = BaseImage(
                id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                name=query_name,
                xys=np.array([]),
                point3D_ids=np.array([]),
            )

        write_images_binary(images, output_path / 'images.bin')
        write_images_text(images, output_path / 'images.txt')

    @staticmethod
    def write_intrinsics_dict_to_colmap_format(
            output_path: Path,
            intrinsics: Dict[str, Tuple[str, int, int, np.ndarray]],
        ):
        """
        Write intrinsics to COLMAP format.
        """
        cameras = {}
        camera_id = 0

        for query_name, (camera_model, w, h, camera_params) in intrinsics.items():
            camera_id += 1
            cameras[camera_id] = Camera(
                id=camera_id,
                model=camera_model,
                width=w,
                height=h,
                params=camera_params,
            )

        write_cameras_binary(cameras, output_path / 'cameras.bin')
        write_cameras_text(cameras, output_path / 'cameras.txt')

    @staticmethod
    def write_poses_text_file_to_colmap_format(
            path_to_poses: Path,
            poses_file: str,
            quaternion_first: bool,
        ):
        """
        Write poses text file to COLMAP format.
        """
        poses = ColmapModelWriter.read_poses_text_file(path_to_poses, poses_file, quaternion_first)
        ColmapModelWriter.write_poses_dict_to_colmap_format(path_to_poses, poses)

    @staticmethod
    def write_intrinsics_text_file_to_colmap_format(
            path_to_intrinsics: Path,
            intrinsics_file: str,
        ):
        """
        Write intrinsics text file to COLMAP format.
        """
        intrinsics = ColmapModelWriter.read_intrinsics_text_file(path_to_intrinsics, intrinsics_file)
        ColmapModelWriter.write_intrinsics_dict_to_colmap_format(path_to_intrinsics, intrinsics)


    @staticmethod
    def write_poses_and_intrinsics_text_files_to_colmap_format(
            path_to_intrinsics: Path,
            path_to_poses: Path,
            poses_file: str,
            intrinsincs_file: str,
            quaternion_first: bool,
        ):
        """
        Write query poses and intrinsics to COLMAP format.
        Uses same id for camera and image.
        """

        ColmapModelWriter.write_intrinsics_text_file_to_colmap_format(path_to_intrinsics, intrinsincs_file)
        ColmapModelWriter.write_poses_text_file_to_colmap_format(path_to_poses, poses_file, quaternion_first)


class ColmapModelConverter:

    """
    Convert COLMAP model data between text and binary formats.
    """
    @staticmethod
    def convert_text_to_binary(
            path_to_text: Path,
            path_to_binary: Path,
            ):
        """
        Convert text to binary files.
        """
        path_to_cameras = path_to_binary / 'cameras.txt'
        print(path_to_cameras)
        if (path_to_text / 'cameras.txt').exists():
            cameras = read_cameras_text(path_to_text / 'cameras.txt')
            if not (path_to_binary / 'cameras.bin').exists():
                write_cameras_binary(cameras, path_to_binary / 'cameras.bin')

        if (path_to_text / 'images.txt').exists():
            images = read_images_text(path_to_text / 'images.txt')
            if not (path_to_binary / 'images.bin').exists():
                write_images_binary(images, path_to_binary / 'images.bin')
        
        if (path_to_text / 'points3D.txt').exists():
            points3D = read_points3D_text(path_to_text / 'points3D.txt')
            if not (path_to_binary / 'points3D.bin').exists():
                write_points3D_binary(points3D, path_to_binary / 'points3D.bin')

    @staticmethod
    def convert_binary_to_text(
            path_to_binary: Path,
            path_to_text: Path,
            ):
        """
        Convert binary to text files.
        """
        if (path_to_binary / 'cameras.bin').exists():
            cameras = read_cameras_binary(path_to_binary / 'cameras.bin')
            if not (path_to_text / 'cameras.txt').exists():
                write_cameras_text(cameras, path_to_text / 'cameras.txt')
        
        if (path_to_binary / 'images.bin').exists():
            images = read_images_binary(path_to_binary / 'images.bin')
            if not (path_to_text / 'images.txt').exists():
                write_images_text(images, path_to_text / 'images.txt')
        
        if (path_to_binary / 'points3D.bin').exists():
            points3D = read_points3D_binary(path_to_binary / 'points3D.bin')
            if not (path_to_text / 'points3D.txt').exists():
                write_points3D_text(points3D, path_to_text / 'points3D.txt')


if __name__ == '__main__':

    pass

    # from data_new import Model, CadModel

    # model = Model('Notre Dame')
    # cad_model = CadModel(model, 'B')

    # poses_file = '25_superglue_aachen_v1_1__20.0_keypoint_clusters_POSELIB+REF_min_10000_max_100000_ref_1.0_0.25_bias_0.0_0.0.txt'

    # ColmapModelWriter.write_query_poses_to_colmap_format(
    #     path_to_query=cad_model.path_to_query,
    #     path_to_poses=cad_model.path_to_meshloc_out,
    #     poses_file=poses_file,
    #     quaternion_first=True,
    # )

    # ColmapModelWriter.write_query_poses_to_colmap_format(
    #     path_to_query=cad_model.path_to_query,
    #     path_to_poses=cad_model.path_to_ground_truth,
    #     poses_file='cam_sfm_poses.txt',
    #     quaternion_first=False,
    # )

    # ColmapModelWriter.write_query_poses_to_colmap_format(
    #     path_to_query=Path('/Users/eric/Developer/meshloc_dataset/aachen_day_night_v11/'),
    #     path_to_poses=Path('/Users/eric/Developer/meshloc_output/aachen_day_night_v11/superglue/experiment_output/'),
    #     poses_file='50_superglue_aachen_v1_1__20.0_keypoint_clusters_POSELIB+REF_min_10000_max_100000_ref_1.0_0.25_bias_0.0_0.0.txt',
    #     quaternion_first=True,
    # )

    # path_to_text = Path('/Users/eric/Downloads/colmap visualization/Aachen/input MeshLoc/')
    # path_to_binary = path_to_text
    # ColmapModelConverter.convert_text_to_binary(path_to_text, path_to_binary)

    # path_to_binary = Path('/Users/eric/Downloads/colmap visualization/Notre Dame B/output queries MeshLoc/')
    # path_to_text = path_to_binary
    # ColmapModelConverter.convert_binary_to_text(path_to_binary, path_to_text)

    '''
    Notre Dame B
    '''

    # output_path = Path('/Users/eric/Documents/Studies/MSc Robotics/Thesis/Evaluation/notre_dame_B/outputs/meshloc_out/patch2pix/')
    # poses_file = '25_patch2pix_aachen_v1_1__20.0_keypoint_clusters_POSELIB+REF_min_10000_max_100000_ref_1.0_0.25_bias_0.0_0.0.txt'
    # intrinsics_file = 'queries.txt'

    # ColmapModelWriter.write_poses_text_file_to_colmap_format(output_path, poses_file, quaternion_first=True)
    # ColmapModelWriter.write_intrinsics_text_file_to_colmap_format(output_path, intrinsics_file)


    '''
    Notre Dame E
    '''

    # output_path = Path('/Users/eric/Documents/Studies/MSc Robotics/Thesis/Evaluation/notre_dame_E/outputs/meshloc_out/patch2pix/')
    # poses_file = '20_patch2pix_aachen_v1_1__20.0_keypoint_clusters_POSELIB+REF_min_10000_max_100000_ref_1.0_0.25_bias_0.0_0.0.txt'
    # intrinsics_file = 'queries.txt'

    # ColmapModelWriter.write_poses_text_file_to_colmap_format(output_path, poses_file, quaternion_first=True)
    # ColmapModelWriter.write_intrinsics_text_file_to_colmap_format(output_path, intrinsics_file)



    '''
    Aachen Day-Night
    - Patch2Pix
    - SuperGlue
    '''
    # output_path = Path('/Users/eric/Developer/meshloc_output/aachen_day_night_v11/patch2pix/experiment_output/')
    # poses_file = '50_patch2pix_aachen_v1_1__20.0_keypoint_clusters_POSELIB+REF_min_10000_max_100000_ref_1.0_0.25_bias_0.0_0.0.txt'
    # intrinsics_file = 'night_time_queries_with_intrinsics_800_basenames.txt'

    # ColmapModelWriter.write_poses_text_file_to_colmap_format(output_path, poses_file, quaternion_first=True)
    # ColmapModelWriter.write_intrinsics_text_file_to_colmap_format(output_path, intrinsics_file)

    # output_path = Path('/Users/eric/Developer/meshloc_output/aachen_day_night_v11/superglue/experiment_output/')
    # poses_file = '50_superglue_aachen_v1_1__20.0_keypoint_clusters_POSELIB+REF_min_10000_max_100000_ref_1.0_0.25_bias_0.0_0.0.txt'
    # intrinsics_file = 'night_time_queries_with_intrinsics_800_basenames.txt'

    # ColmapModelWriter.write_poses_text_file_to_colmap_format(output_path, poses_file, quaternion_first=True)
    # ColmapModelWriter.write_intrinsics_text_file_to_colmap_format(output_path, intrinsics_file)