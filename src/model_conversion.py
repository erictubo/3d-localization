import numpy as np
from math import pi
from pathlib import Path
from typing import Tuple, Dict, List
from scipy.spatial.transform import Rotation
from pyquaternion import Quaternion
from transformations import decompose_matrix, compose_matrix
from matplotlib import pyplot as plt
import OpenEXR
import Imath

from colmap.read_write_model import (
    Camera, write_cameras_binary, write_cameras_text,
    BaseImage, write_images_binary, write_images_text,
    Point3D, write_points3D_binary, write_points3D_text,
)


def convert_pose_to_matrix(pose: np.ndarray) -> np.ndarray:
    """
    Convert pose vector to transformation matrix.
    Format: scalar-first (px, py, pz, qw, qx, qy, qz)
    """
    assert pose.shape == (7,), pose.shape
    t = pose[:3]
    q = Quaternion(pose[3:])
    R = q.rotation_matrix
    T = np.eye(4)
    T[0:3, :] = np.c_[R, t]

    return T

def convert_matrix_to_pose(T: np.ndarray) -> np.ndarray:
    """
    Convert transformation matrix to pose vector.
    Format: scalar-first (px, py, pz, qw, qx, qy, qz)
    """
    assert T.shape == (4,4), T.shape
    t = T[:3, 3]
    R = T[:3, :3]
    q = Quaternion(matrix=R).q
    pose = np.append(t, q)

    return pose


class ModelConversion:
    """
    Convert between formats of CAD and SFM (COLMAP) models.
    """
    def __init__(
            self,
            path_to_ground_truth: Path,
            path_to_database: Path,
            images_prefix: str = 'images/',
            poses_prefix: str = 'poses/',
            depth_prefix: str = 'depth/',
            ):
        
        self.path_to_ground_truth = path_to_ground_truth
        self.T_cad_sfm = self.read_registration_matrix(path_to_ground_truth / 'T_cad_sfm.txt')
        self.T_sfm_cad = np.linalg.inv(self.T_cad_sfm)

        self.path_to_ground_truth = path_to_ground_truth
        self.path_to_database = path_to_database

        self.path_to_poses = path_to_database / poses_prefix
        self.path_to_images = path_to_database / images_prefix
        self.path_to_depth = path_to_database / depth_prefix

        self.image_names = self.read_image_names(self.path_to_images)
        
    @staticmethod
    def read_image_names(file_path) -> List[str]:
        file_names = [f.name for f in file_path.iterdir() if (f.is_file() and not f.name.startswith('.'))]
        image_names = [name for name in file_names if name.split('.')[-1] in ['png', 'jpg', 'jpeg']]
        return image_names

    @staticmethod
    def read_registration_matrix(file_path: Path) -> np.ndarray:
        T_cad_sfm = np.loadtxt(file_path)
        assert T_cad_sfm.shape == (4,4), T_cad_sfm.shape

        scale, shear, angles, translate, perspective = decompose_matrix(T_cad_sfm)
        q_cad_sfm = Quaternion(matrix=Rotation.from_euler('xyz', angles).as_matrix()).inverse
        pose_cad_sfm = np.hstack([translate, q_cad_sfm.q])
        s_cad_sfm = sum(scale) / len(scale)
        print('SFM pose (CAD frame): ', pose_cad_sfm)
        print('SFM vs. CAD scale: ', s_cad_sfm)

        return T_cad_sfm
    

    """
    FRAME CONVERSION:
    """
    
    @staticmethod
    def reverse_camera_matrix_for_blender(T: np.ndarray) -> np.ndarray:
        """
        Rotate camera matrix 180 degrees about the x-axis to match Blender camera (facing -z direction).
        """
        assert T.shape == (4,4), T.shape
        T_rev = compose_matrix(None, None, [np.pi, 0, 0], [0, 0, 0]) @ T
        return T_rev

    def reverse_camera_pose_for_blender(self, pose: np.ndarray) -> np.ndarray:
        """
        Rotate camera pose 180 degrees about the x-axis to match Blender camera (facing -z direction).
        """
        assert pose.shape == (7,), pose.shape
        T = convert_pose_to_matrix(pose)
        T_rev = self.reverse_camera_matrix_for_blender(T)
        pose_rev = convert_matrix_to_pose(T_rev)
        return pose_rev

    def transform_pose_from_colmap_to_cad_format(
            self,
            pose_cam_sfm: np.ndarray,
            to_blender_format: bool,
            ) -> np.ndarray:
        """
        Transform pose in COLMAP to CAD format.
        Optional: reverse pose to Blender format (camera facing -z direction).
        """

        # SFM in CAM frame
        T_cam_sfm = convert_pose_to_matrix(pose_cam_sfm)

        if to_blender_format:
            T_cam_sfm = self.reverse_camera_matrix_for_blender(T_cam_sfm)

        # CAM in SFM frame
        T_sfm_cam = np.linalg.inv(T_cam_sfm)

        # CAM in CAD frame
        T_cad_cam = self.T_cad_sfm @ T_sfm_cam
        scale, shear, angles, translate, perspective = decompose_matrix(T_cad_cam)
        T_cad_cam = compose_matrix(None, None, angles, translate)
        pose_cad_cam = convert_matrix_to_pose(T_cad_cam)

        pose_cad_cam[:3] = pose_cad_cam[:3] / scale

        return pose_cad_cam
    
    def transform_pose_from_cad_to_colmap_format(
            self,
            pose_cad_cam: np.ndarray,
            from_blender_format: bool,
            ) -> np.ndarray:
        """
        Transform pose in CAD to COLMAP format.
        Optional: reverse camera pose from Blender format (facing -z direction).
        """
        
        # CAM in CAD frame
        T_cad_cam = convert_pose_to_matrix(pose_cad_cam)

        # CAM in SFM frame
        T_sfm_cam = self.T_sfm_cad @ T_cad_cam

        # SFM in CAM frame
        T_cam_sfm = np.linalg.inv(T_sfm_cam)
        scale, shear, angles, translate, perspective = decompose_matrix(T_cam_sfm)
        T_cam_sfm = compose_matrix(None, None, angles, translate)

        if from_blender_format:
            T_cam_sfm = self.reverse_camera_matrix_for_blender(T_cam_sfm)

        pose_cam_sfm = convert_matrix_to_pose(T_cam_sfm)
        
        pose_cam_sfm[:3] = pose_cam_sfm[:3] / scale

        return pose_cam_sfm


    """
    WRITE POSES & INTRINSICS
    """
    
    def convert_render_intrinsics_and_poses_to_colmap_format(self, from_blender_format: bool):
        """
        Convert render intrinsics and poses to COLMAP format.
        Assumption: all images have the same intrinsics.
        """

        # Read intrinsics from database
        with open(self.path_to_database / 'intrinsics.txt', 'r') as file:
            lines = file.readlines()
            w, h, f_mm = [float(x) for x in lines[0].split()]
            w, h = int(w), int(h)
            fx = fy = f_mm * w / 36
            cx, cy = w/2, h/2

        camera = Camera(
            id=1,
            model='PINHOLE',
            width=w,
            height=h,
            params=np.array([fx, fy, cx, cy]),
        )

        cameras = {camera.id: camera}
        write_cameras_binary(cameras, self.path_to_database / 'cameras.bin')
        write_cameras_text(cameras, self.path_to_database / 'cameras.txt')

        images = {}
        for i, image_name in enumerate(self.image_names):

            image_id = i+1

            # Read pose from database
            pose_cad_cam = np.loadtxt(self.path_to_poses / (image_name.split('.')[0] + '.txt'))
            assert pose_cad_cam.shape == (7,), pose_cad_cam.shape

            # Convert pose from CAD to COLMAP format
            pose_cam_sfm = self.transform_pose_from_cad_to_colmap_format(
                pose_cad_cam=pose_cad_cam,
                from_blender_format=from_blender_format,
            )
            tvec, qvec = pose_cam_sfm[:3], pose_cam_sfm[3:]

            # Use empty 2D-3D correspondences for completeness
            xys = np.array([])
            point3D_ids = np.array([])

            images[image_id] = BaseImage(
                id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera.id,
                name=image_name,
                xys=xys,
                point3D_ids=point3D_ids,
            )
        
        write_images_binary(images, self.path_to_database / 'images.bin')
        write_images_text(images, self.path_to_database / 'images.txt')

        points3d = {}
        # Read bounding box coordinates from database
        with open(self.path_to_database / 'bounding_box.txt', 'r') as file:
            lines = file.readlines()
            for i, line in enumerate(lines):
                line.strip()
                xyz_cad = np.array([float(x) for x in line.split()])
                xyz_sfm = (self.T_sfm_cad @ np.append(xyz_cad, 1))[:3]

                print(f'Point {i+1}: {xyz_cad} (CAD) -> {xyz_sfm} (SFM)')

                points3d[i+1] = Point3D(
                    id=i+1,
                    xyz=xyz_sfm,
                    rgb=np.array([0, 0, 0]),
                    error=0,
                    image_ids=np.array([]),
                    point2D_idxs=np.array([]),
                )

        write_points3D_binary(points3d, self.path_to_database / 'points3D.bin')
        write_points3D_text(points3d, self.path_to_database / 'points3D.txt')



    """
    DEPTH CONVERSION:
    """

    @staticmethod
    def convert_depth_map_from_exr_to_npz(path_to_depth: Path, name: str):
        """
        Convert EXR depth maps to NPZ format.
        """
        name = name.split('.')[0]
        if not name.endswith('_depth'):
            name += '_depth'
        depth_name = name + '.exr'

        file = OpenEXR.InputFile(str(path_to_depth / depth_name))

        # Get the header and data window
        dw = file.header()['dataWindow']
        size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

        # Read the depth channel
        depth_str = file.channel('V', Imath.PixelType(Imath.PixelType.FLOAT))
        depth = np.frombuffer(depth_str, dtype=np.float32)
        depth = depth.reshape(size[1], size[0])

        # Set background to zero
        depth = np.where(depth < 1000, depth, 0)

        depth_values_dict = {}
        depth_values_dict['depth'] = depth        
        np.savez_compressed(path_to_depth / (name + '.npz'), **depth_values_dict)

    def convert_depth_maps_from_exr_to_npz(self):
        """
        Convert all EXR depth maps to NPZ format.
        """
        for image_name in self.image_names:
            self.convert_depth_map_from_exr_to_npz(self.path_to_depth, image_name)

    @staticmethod
    def visualize_depth_map(path_to_depth: Path, name: str, out: str = 'show'):
        """
        Visualize npy/npz depth map with colors according to depth values and a legend.
        Goal: to check if depth map is correct.
        """

        name = name.split('.')[0]
        if not name.endswith('_depth'):
            name += '_depth'
        depth_name = name + '.npz'

        depth_map = np.load(path_to_depth / depth_name)['depth']

        # Create a custom colormap with white for zero values
        cmap = plt.get_cmap('viridis').copy()
        cmap.set_bad(color='white')
        
        # Create a masked array, masking zero values
        masked_depth_map = np.ma.masked_where(depth_map == 0, depth_map)
        
        fig, ax = plt.subplots(figsize=(10, 8))

        # Create a color-coded image of the depth map
        im = ax.imshow(masked_depth_map, cmap='viridis')

        # Reverse the y-axis
        # ax.invert_yaxis()
        
        # Add a colorbar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Depth', rotation=270, labelpad=15)
        
        # Set title and labels
        ax.set_title('Depth Map Visualization')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
        if out == 'show':
            plt.show()
        elif out == 'save':
            plt.savefig(path_to_depth / f'{name}_depth.png')
            plt.close()
        else:
            raise ValueError(f"Invalid output option: {out}")
        
    def visualize_depth_maps(self):
        """
        Create depth images from depth maps and save them.
        """
        for image_name in self.image_names:
            self.visualize_depth_map(self.path_to_depth, image_name, out='save')