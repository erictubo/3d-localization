import numpy as np
from pathlib import Path
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation
from transformations import decompose_matrix, compose_matrix
from typing import Tuple, Dict, List

from transformation import pose_to_matrix, matrix_to_pose, invert_pose, invert_matrix


class CadData:

    def __init__(
            self,
            path_to_ground_truth: Path,
            path_to_database: Path,
            poses_prefix: str = 'poses/',
            ):
        
        self.path_to_ground_truth = path_to_ground_truth
        self.path_to_database = path_to_database
        self.path_to_database_poses = path_to_database / poses_prefix
        
        self.T_cad_sfm = self.read_registration_matrix()
        
    def read_registration_matrix(self, file_name: str = 'T_cad_sfm.txt') -> np.ndarray:
        T_cad_sfm = np.loadtxt(self.path_to_ground_truth / file_name)
        assert T_cad_sfm.shape == (4,4), self.T_cad_sfm.shape

        scale, shear, angles, translate, perspective = decompose_matrix(T_cad_sfm)
        q_cad_sfm = Quaternion(matrix=Rotation.from_euler('xyz', angles).as_matrix()).inverse
        pose_cad_sfm = np.hstack([translate, q_cad_sfm.q])
        s_cad_sfm = sum(scale) / len(scale)
        print('SFM pose (CAD frame): ', pose_cad_sfm)
        print('SFM vs. CAD scale: ', s_cad_sfm)

        return T_cad_sfm
    
    def read_db_pose(self, image_name: str) -> np.ndarray:
        """
        Get pose of database image from render output text file.
        Format: scalar-first (px, py, pz, qw, qx, qy, qz)
        """
        pose_name = image_name.replace('.png', '.txt')
        pose = np.loadtxt(self.path_to_database_poses / pose_name)
        assert pose.shape == (7,), pose.shape

        return pose

    def convert_query_pose_to_cad_frame(self, pose_cam_sfm: np.ndarray) -> np.ndarray:
        """
        Get pose of query image in CAD frame.
        """
        # SFM in CAM frame
        # rotate 180 degrees about the x-axis to match Blender camera (facing -z direction)
        T_cam_sfm = compose_matrix(None, None, [np.pi, 0, 0], [0, 0, 0]) @ pose_to_matrix(pose_cam_sfm)

        # CAM in SFM frame
        T_sfm_cam = np.linalg.inv(T_cam_sfm)
        # pose_sfm_cam = matrix_to_pose(T_sfm_cam)
        # print('CAM pose (SFM frame): ', pose_sfm_cam)

        # CAM in CAD frame
        T_cad_cam = self.T_cad_sfm @ T_sfm_cam
        scale, shear, angles, translate, perspective = decompose_matrix(T_cad_cam)
        T_cad_cam = compose_matrix(None, None, angles, translate)
        pose_cad_cam = matrix_to_pose(T_cad_cam)
        # print('CAM pose (CAD frame): ', pose_cad_cam)

        return pose_cad_cam