import numpy as np
from pathlib import Path
from typing import Tuple
from math import pi

from hloc.utils import read_write_model


class ColmapData:
    """
    Data retrieval class for COLMAP SfM ground truth database.
    """

    def __init__(self, path_to_model: Path):

        self.path_to_model = path_to_model

        self.images_db = read_write_model.read_images_binary(path_to_model / 'dense/sparse/' / 'images.bin')
        self.cameras_db = read_write_model.read_cameras_binary(path_to_model / 'dense/sparse/' / 'cameras.bin')
        # self.points3D_db = read_write_model.read_points3d_binary(path_to_model / 'dense/sparse/' / 'points3D.bin')

        # images_db[image_id] = Image(id, qvec, tvec, camera_id, name, xys, point3D_ids)
        
        # cameras[camera_id] = Camera(id, model, width, height, params)

        # points3D[point3D_id] = Point3D(id, xyz, rgb, error, image_ids, point2D_idxs)


    def get_image_id(self, image_name: str) -> int:
        """
        Get image ID from image name.
        """
        image_name = image_name.split('/')[-1]
        image_id = [k for k, v in self.images_db.items() if v.name == image_name][0]
        return image_id
    
    def get_pose(self, image_id: int) -> np.ndarray:
        """
        Get pose of image from database.
        """
        sfm_image = self.images_db[image_id]
        pose = np.concatenate([sfm_image.tvec, sfm_image.qvec])
        assert pose.shape == (7,), pose.shape
        return pose

    # def get_pose_from_name(self, image_name: str) -> np.ndarray:
    #     """
    #     Get pose of image from database.
    #     """
    #     image_id = self.get_query_sfm_id(image_name, self.images_db)
    #     return self.get_pose_from_id(image_id)
    
    def get_intrinsics(self, image_id: int) -> Tuple[str, np.ndarray, int, int]:
        """
        Get intrinsics of image from database.
        Output: camera model, camera parameters, image width, image height
        """
        sfm_image = self.images_db[image_id]
        camera_id = sfm_image.camera_id

        sfm_camera = self.cameras_db[camera_id]

        camera_model: str = sfm_camera.model
        camera_params: np.ndarray = sfm_camera.params
        w: int = sfm_camera.width
        h: int = sfm_camera.height
        
        return camera_model, camera_params, w, h

    # def get_intrinsics_from_name(self, image_name: str) -> Tuple[str, np.ndarray, int, int]:
    #     """
    #     Get intrinsics of query image from COLMAP SfM ground truth database.
    #     Output: camera model, camera parameters, image width, image height
    #     """
    #     image_id = self.get_query_sfm_id(image_name, self.images_db)
    #     return self.get_intrinsics_from_id(image_id)