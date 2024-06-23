import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List
from math import pi

from hloc.utils import read_write_model


class SfmData:
    """
    Data retrieval class for COLMAP SfM ground truth database:
    - images[image_id] = Image(id, qvec, tvec, camera_id, name, xys, point3D_ids)   
    - cameras[camera_id] = Camera(id, model, width, height, params)
    - points3D[point3D_id] = Point3D(id, xyz, rgb, error, image_ids, point2D_idxs)
    """

    def __init__(self, path_to_data: Path):

        self.path_to_data = path_to_data
        self.cameras, self.images, self.points3d = read_write_model.read_model(path_to_data)
        
    def get_image_id(self, image_name: str) -> int:
        """
        Get image ID from image name.
        """
        image_name = image_name.split('/')[-1]
        image_id = [k for k, v in self.images.items() if v.name == image_name][0]
        return image_id
    
    def get_pose(self, image_id: int) -> np.ndarray:
        """
        Get pose of image from database.
        """
        sfm_image = self.images[image_id]
        pose = np.concatenate([sfm_image.tvec, sfm_image.qvec])
        assert pose.shape == (7,), pose.shape
        return pose
    
    def get_intrinsics(self, image_id: int) -> Tuple[str, int, int, np.ndarray]:
        """
        Get intrinsics of image from database.
        Output: camera model, camera parameters, image width, image height
        """
        sfm_image = self.images[image_id]
        camera_id = sfm_image.camera_id

        sfm_camera = self.cameras[camera_id]

        camera_model: str = sfm_camera.model
        w: int = sfm_camera.width
        h: int = sfm_camera.height
        camera_params: np.ndarray = sfm_camera.params
        
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
                query_id = self.get_image_id(query_name)
                (camera_model, w, h, camera_params) = self.get_intrinsics(query_id)
                f.write(f"{query_name} {camera_model} {w} {h} {' '.join(map(str, camera_params))}\n")

    def write_query_poses_text_file(
            self,
            path_to_output: Path,
            query_names: List[str],
            file_name: str = 'cam_sfm_poses.txt',
            ):
        """
        Write ground truth text file in outputs directory (for use with MeshLoc).
        Format in each line: query_name, pose
        """
        if not file_name.endswith('.txt'):
            file_name += '.txt'
        with open(path_to_output / file_name, 'w') as f:
            for query_name in query_names:
                query_id = self.get_image_id(query_name)
                pose = self.get_pose(query_id)
                f.write(f"{query_name} {' '.join(map(str, pose))}\n")