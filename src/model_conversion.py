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

from colmap.read_write_model import *


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


def reverse_camera_pose_for_blender(pose: np.ndarray, pose_frame: str) -> np.ndarray:
    """
    Rotate camera matrix 180 degrees about the x-axis to match Blender camera (facing -z direction).
    Params:
    - pose: [px, py, pz, qx, qy, qz] OR 4x4 homogeneous transformation matrix
    - pose_frame: CAM (faster) or in another frame (will do an inversion for the correct axis)
    """
    if pose.shape == (7,):
        T = convert_pose_to_matrix(pose)
    elif pose.shape == (4,4):
        T = pose
    else:
        raise ValueError(f'Pose shape {pose.shape} not valid - should be (7,) or (4,4).')
    
    if pose_frame.lower()[:3] == 'cam':
        # rotate about camera's x-axis by 180 degrees
        T_rev = compose_matrix(None, None, [np.pi, 0, 0], [0, 0, 0]) @ T

    else:
        # first invert the matrix, then rotate about x-axis by 180 degrees, then invert again
        T_inv = np.linalg.inv(T)
        T_inv_rev = compose_matrix(None, None, [np.pi, 0, 0], [0, 0, 0]) @ T_inv
        T_rev = np.linalg.inv(T_inv_rev)

    if pose.shape == (7,):
        pose_rev = convert_matrix_to_pose(T_rev)
    elif pose.shape == (4,4):
        pose_rev = T_rev

    assert pose.shape == pose_rev.shape, (pose.shape, pose_rev.shape)
    return pose_rev


class ModelConversion:
    """
    Convert between formats of CAD and SFM (COLMAP) models.
    - transform poses between CAD and COLMAP formats.
    - convert depth maps from EXR to NPZ format.
    - visualize depth maps.
    """
    def __init__(
            self,
            path_to_ground_truth: Path,
            path_to_database: Path = None,
            images_prefix: str = 'images/',
            intrinsics_prefix: str = 'intrinsics/',
            poses_prefix: str = 'poses/',
            depth_prefix: str = 'depth/',
            scene_coordinates_prefix: str = 'scene_coordinates/',
            ):
        
        self.path_to_ground_truth = path_to_ground_truth
        # self.T_cad_sfm, self.s_cad_sfm = self.read_registration_data(path_to_ground_truth / 'T_cad_sfm.txt')
        # self.T_sfm_cad = np.linalg.inv(self.T_cad_sfm)
        self.T_sfm_cad, self.s_sfm_cad = self.read_registration_data(path_to_ground_truth / 'T_sfm_cad.txt')
        self.T_cad_sfm = np.linalg.inv(self.T_sfm_cad)
        self.s_cad_sfm = 1/self.s_sfm_cad

        self.path_to_ground_truth = path_to_ground_truth

        if path_to_database:
            self.path_to_database = path_to_database

            self.path_to_poses = path_to_database / poses_prefix
            self.path_to_images = path_to_database / images_prefix
            self.path_to_depth = path_to_database / depth_prefix
            self.path_to_intrinsics = path_to_database / intrinsics_prefix
            self.path_to_scene_coordinates = path_to_database / scene_coordinates_prefix

            self.image_names = self.read_image_names(self.path_to_images)
        
    @staticmethod
    def read_image_names(file_path) -> List[str]:
        file_names = [f.name for f in file_path.iterdir() if (f.is_file() and not f.name.startswith('.'))]
        image_names = [name for name in file_names if name.split('.')[-1] in ['png', 'jpg', 'jpeg']]
        return image_names

    @staticmethod
    def read_registration_data(file_path: Path) -> np.ndarray:
        T = np.loadtxt(file_path)
        assert T.shape == (4,4), T.shape

        scale, shear, angles, translate, perspective = decompose_matrix(T)
        q = Quaternion(matrix=Rotation.from_euler('xyz', angles).as_matrix()).inverse
        pose = np.hstack([translate, q.q])
        scale = np.average(scale)
        print('Pose:', pose)
        print('Scale:', scale)

        return T, scale
    

    """
    FRAME CONVERSION:
    """

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
        if pose_cam_sfm.shape == (7,):
            T_cam_sfm = convert_pose_to_matrix(pose_cam_sfm)
        elif pose_cam_sfm.shape == (4,4):
            T_cam_sfm = pose_cam_sfm
        else:
            raise ValueError(f'Pose shape {pose_cam_sfm.shape} not valid - should be (7,) or (4,4).')

        if to_blender_format:
            T_cam_sfm = reverse_camera_pose_for_blender(T_cam_sfm, pose_frame='camera')

        # CAM in SFM frame
        T_sfm_cam = np.linalg.inv(T_cam_sfm)

        # CAM in CAD frame
        T_cad_cam = self.T_cad_sfm @ T_sfm_cam
        scale, shear, angles, translate, perspective = decompose_matrix(T_cad_cam)
        T_cad_cam = compose_matrix(None, None, angles, translate)

        if pose_cam_sfm.shape == (7,):
            pose_cad_cam = convert_matrix_to_pose(T_cad_cam)
        elif pose_cam_sfm.shape == (4,4):
            pose_cad_cam = T_cad_cam

        assert pose_cam_sfm.shape == pose_cad_cam.shape, (pose_cam_sfm.shape, pose_cad_cam.shape)
        return pose_cad_cam
    
    def transform_poses_from_colmap_to_cad_format(
            self,
            poses_cam_sfm: Dict[str, np.ndarray],
            to_blender_format: bool,
            ) -> Dict[str, np.ndarray]:
        """
        Transform poses in COLMAP to CAD format.
        Optional: reverse poses to Blender format (camera facing -z direction).
        """
        poses_cad_cam = {}
        for query_name, pose_cam_sfm in poses_cam_sfm.items():
            pose_cad_cam = self.transform_pose_from_colmap_to_cad_format(
                pose_cam_sfm=pose_cam_sfm,
                to_blender_format=to_blender_format,
            )
            poses_cad_cam[query_name] = pose_cad_cam
        return poses_cad_cam
    
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
        if pose_cad_cam.shape == (7,):
            T_cad_cam = convert_pose_to_matrix(pose_cad_cam)
        elif pose_cad_cam.shape == (4,4):
            T_cad_cam = pose_cad_cam
        else:
            raise ValueError(f'Pose shape {pose_cad_cam.shape} not valid - should be (7,) or (4,4).')

        if from_blender_format:
            T_cad_cam = reverse_camera_pose_for_blender(T_cad_cam, 'cad')

        # CAM in SFM frame
        T_sfm_cam = self.T_sfm_cad @ T_cad_cam

        # SFM in CAM frame
        T_cam_sfm = np.linalg.inv(T_sfm_cam)
        scale, shear, angles, translate, perspective = decompose_matrix(T_cam_sfm)
        T_cam_sfm = compose_matrix(None, None, angles, translate)

        pose_cam_sfm = convert_matrix_to_pose(T_cam_sfm)
        
        pose_cam_sfm[:3] = pose_cam_sfm[:3] / scale

        if pose_cad_cam.shape == (4,4):
            pose_cam_sfm = T_cam_sfm

        assert pose_cad_cam.shape == pose_cam_sfm.shape, (pose_cad_cam.shape, pose_cam_sfm.shape)
        return pose_cam_sfm
    
    def transform_poses_from_cad_to_colmap_format(
            self,
            poses_cad_cam: Dict[str, np.ndarray],
            from_blender_format: bool,
            ) -> Dict[str, np.ndarray]:
        """
        Transform poses in CAD to COLMAP format.
        Optional: reverse camera poses from Blender format (facing -z direction).
        """
        poses_cam_sfm = {}
        for query_name, pose_cad_cam in poses_cad_cam.items():
            pose_cam_sfm = self.transform_pose_from_cad_to_colmap_format(
                pose_cad_cam=pose_cad_cam,
                from_blender_format=from_blender_format,
            )
            poses_cam_sfm[query_name] = pose_cam_sfm
        return poses_cam_sfm


    """
    WRITE POSES & INTRINSICS
    """
    
    def convert_render_intrinsics_and_poses_to_colmap_format(self, from_blender_format: bool):
        """
        Convert render intrinsics and poses to COLMAP format.
        Assumption: fx=fy (only used along largest dimension).
        """

        cameras = {}
        images = {}

        camera_id = 1
        previous_intrinsics = {} # dictionary of unique previous intrinsics with camera_id

        for i, image_name in enumerate(self.image_names):

            image_id = i+1
            name = image_name.split('.')[0]

            # 1. INTRINSICS

            # Read intrinsics from file (w, h, f, f_unit: str, cx, cy)
            # read string from file, not as numpy array
            file = self.path_to_intrinsics / (name + '.txt')
            intrinsics = file.read_text()
        
            # If intrinsics are new, add a new camera
            if intrinsics not in previous_intrinsics.keys():
                previous_intrinsics[intrinsics] = camera_id

                # Extract and convert intrinsics
                intrinsics = intrinsics.strip().split()
                assert len(intrinsics) == 6, print(f'intrinsics {intrinsics} of length {len(intrinsics)} != 6')
                w, h, f, f_unit, cx, cy = int(intrinsics[0]), int(intrinsics[1]), float(intrinsics[2]), intrinsics[3], float(intrinsics[4]), float(intrinsics[5])

                if f_unit.upper() == 'MM':
                    fx = fy = f * max(w,h) / 36
                elif f_unit.upper() == 'PIX':
                    fx = fy = f
                else:
                    raise ValueError(f'Focal length unit {f_unit} not implemented.')
            
                # Create camera
                camera = Camera(
                    id=camera_id,
                    model='PINHOLE',
                    width=w,
                    height=h,
                    params=np.array([fx, fy, cx, cy]),
                )

                cameras[camera_id] = camera
                camera_id += 1

            else:
                camera_id = previous_intrinsics[intrinsics]
                assert camera_id in cameras.keys(), print(camera_id, cameras.keys())


            # 2. POSE

            # Read pose from file
            pose_cad_cam = np.loadtxt(self.path_to_poses / (name + '.txt'))
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

        write_cameras_binary(cameras, self.path_to_database / 'cameras.bin')
        write_cameras_text(cameras, self.path_to_database / 'cameras.txt')
        
        write_images_binary(images, self.path_to_database / 'images.bin')
        write_images_text(images, self.path_to_database / 'images.txt')


        # points3d = {}
        # # Read bounding box coordinates from database
        # with open(self.path_to_database / 'bounding_box.txt', 'r') as file:
        #     lines = file.readlines()
        #     for i, line in enumerate(lines):
        #         line.strip()
        #         xyz_cad = np.array([float(x) for x in line.split()])
        #         xyz_sfm = (self.T_sfm_cad @ np.append(xyz_cad, 1))[:3]

        #         print(f'Point {i+1}: {xyz_cad} (CAD) -> {xyz_sfm} (SFM)')

        #         points3d[i+1] = Point3D(
        #             id=i+1,
        #             xyz=xyz_sfm,
        #             rgb=np.array([0, 0, 0]),
        #             error=0,
        #             image_ids=np.array([]),
        #             point2D_idxs=np.array([]),
        #         )

        # write_points3D_binary(points3d, self.path_to_database / 'points3D.bin')
        # write_points3D_text(points3d, self.path_to_database / 'points3D.txt')



    """
    DEPTH CONVERSION:
    """

    @staticmethod
    def convert_depth_map_from_exr_to_npz(
            path_to_depth: Path,
            name: str,
            scale: float = None,
        ):
        """
        Convert EXR depth maps to NPZ format.
        """
        name = name.split('.')[0]
        if os.path.exists(path_to_depth / (name + '.npz')):
            print(f"NPZ depth map {name} already exists, skipping")
            return
        
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

        # scale depth values
        if scale:
            depth = depth / scale

        depth_values_dict = {}
        depth_values_dict['depth'] = depth        
        np.savez_compressed(path_to_depth / (name + '.npz'), **depth_values_dict)

    def convert_depth_maps_from_exr_to_npz(self, format: str):
        """
        Convert all EXR depth maps to NPZ format.
        """
        if format.lower() == 'cad':
            scale = None
        if format.lower() == 'sfm':
            scale = self.s_cad_sfm
            
        for image_name in self.image_names:
            self.convert_depth_map_from_exr_to_npz(self.path_to_depth, image_name, scale)

    """
    SCENE COORDINATES:
    """

    @staticmethod
    def transform_depth_to_scene_coordinate_map(
            depth_map: np.ndarray,
            camera_intrinsics: np.ndarray,
            pose: np.ndarray,
            pose_format: str,
            format: str,
        ) -> np.ndarray:
        """
        Convert depth map to scene coordinates.
        """

        assert pose.shape == (7,), pose.shape

        if pose_format.lower() == 'cam_sfm':
            pose_cam_sfm = pose
            T_cam_sfm = convert_pose_to_matrix(pose_cam_sfm)
            T_sfm_cam = np.linalg.inv(T_cam_sfm)
        
        elif pose_format.lower() == 'cad_cam':
            pose_cad_cam = pose
            T_cad_cam = convert_pose_to_matrix(pose_cad_cam)
        
        elif pose_format.lower() == 'cad_cam blender':
            pose_cad_cam_blender = pose
            T_cad_cam_blender = convert_pose_to_matrix(pose_cad_cam_blender)
            T_cad_cam = reverse_camera_pose_for_blender(T_cad_cam_blender, format='cad')

        else:
            raise ValueError(f'Pose format {pose_format} not implemented.')

        fx, fy, cx, cy = camera_intrinsics

        x, y = np.meshgrid(np.arange(depth_map.shape[1]), np.arange(depth_map.shape[0]))
        x = (x - cx) * depth_map / fx
        y = (y - cy) * depth_map / fy
        z = depth_map

        ones = np.ones_like(z)
        ones = np.where(z == 0, 0, ones)

        points_cam = np.stack([x, y, z, ones], axis=-1)
        points_cam = points_cam.reshape(-1, 4)

        if format.lower() == 'sfm':
            points_sfm = points_cam @ T_sfm_cam.T
            points_sfm = points_sfm[:, :3]
            points_sfm = points_sfm.reshape(depth_map.shape[0], depth_map.shape[1], 3)
            assert points_sfm.shape == depth_map.shape + (3,), points_sfm.shape

            return points_sfm
        
        elif format.lower() == 'cad':
            points_cad = points_cam @ T_cad_cam.T
            points_cad = points_cad[:, :3]
            points_cad = points_cad.reshape(depth_map.shape[0], depth_map.shape[1], 3)
            assert points_cad.shape == depth_map.shape + (3,), points_cad.shape

            return points_cad




    def convert_depth_to_scene_coordinate_maps(self, format: str):
        """
        Convert all depth maps to scene coordinates and save as NPZ files.
        """

        if not self.path_to_scene_coordinates.exists():
            self.path_to_scene_coordinates.mkdir()


        # OPTION A: read poses & intrinsics from COLMAP files (images.txt, cameras.txt)
        if format.lower() == 'sfm':
            pose_format = 'cam_sfm'
            images = read_images_text(self.path_to_database / 'images.txt')
            cameras = read_cameras_text(self.path_to_database / 'cameras.txt')

        # OPTION B: read poses & intrinsics from Blender files (poses, intrinsics)
        elif format.lower() == 'cad':
            pose_format = 'cad_cam'
        
        else:
            raise ValueError(f'Format {format} not implemented.')


        for image_name in self.image_names:
            name = image_name.split('.')[0]
            if os.path.exists(self.path_to_scene_coordinates / (name + '.npz')):
                print(f"Scene coordinate map {name} already exists, skipping")
                return

            if format.lower() == 'sfm':
                image_id = [k for k, v in images.items() if v.name == image_name][0]
                image = images[image_id]
                camera = cameras[image.camera_id]

                assert camera.model == 'PINHOLE', print(f'Camera model {camera.model} not implemented.')
                camera_intrinsics = camera.params # format: [fx, fy, cx, cy]
                pose = np.append(image.tvec, image.qvec)

            elif format.lower() == 'cad':
                # TODO: read camera_intrinsics from file
                intrinsics_file = self.path_to_intrinsics / (name + '.txt')
                w, h, f, f_unit, cx, cy = intrinsics_file.read_text().strip().split()
                w, h, f, cx, cy = int(w), int(h), float(f), float(cx), float(cy)
                
                if f_unit == 'MM':
                    # convert focal length from mm to pixels
                    fx = fy = f * max(w, h) / 36
                else:
                    raise NotImplementedError(f"Unit {f_unit} not implemented")
                
                camera_intrinsics = [fx, fy, cx, cy]

                pose_file = self.path_to_poses / (name + '.txt')
                pose_blender = np.loadtxt(pose_file)
                pose = reverse_camera_pose_for_blender(pose_blender, 'cad')

            depth_name = name + '.npz'
            depth_map = np.load(self.path_to_depth / depth_name)['depth']

            scene_coordinates = self.transform_depth_to_scene_coordinate_map(depth_map, camera_intrinsics, pose, pose_format, format)

            scene_coordinates_dict = {}
            scene_coordinates_dict['scene_coordinates'] = scene_coordinates
            np.savez_compressed(self.path_to_scene_coordinates / (name + '.npz'), **scene_coordinates_dict)


if __name__ == '__main__':

    T_notre_dame = np.array([
        [-0.04308, -0.07366, -0.0008805, -1.525],
        [0.0245, -0.01336, -0.08065, 4.145],
        [0.06947, -0.04097, 0.02789, 10.74],
        [0, 0, 0, 1]
    ])

    T_reichstag = np.array([
        [-0.0004522, -0.06534, 0.0007033, 0.4911],
        [0.00916, -0.0007597, -0.06469, 1.748],
        [0.06469, -0.0003491, 0.009164, 12.37],
        [0, 0, 0, 1]
    ])

    T_st_peters = np.array([
        [-0.008938, 0.04505, -4.739e-05, 7.153],
        [-0.01353, -0.002731, -0.04381, 1.885],
        [-0.04297, -0.008511, 0.01381, 5.6],
        [0, 0, 0, 1]
    ])

    for T, name in zip([T_notre_dame, T_reichstag, T_st_peters], ['Notre Dame', 'Reichstag', 'St Peters Square']):
        scale, shear, angles, translate, perspective = decompose_matrix(T)
        q = Quaternion(matrix=Rotation.from_euler('xyz', angles).as_matrix()).inverse
        pose = np.hstack([translate, q.q])
        scale = np.average(scale)
        print(name)
        print('Scale:', 1/scale)

    

    # path_to_database = Path('/Users/eric/Documents/Studies/MSc Robotics/Thesis/Evaluation/notre_dame_B/inputs/database/')

    # path_to_depth = path_to_database / 'depth/'
    # path_to_poses = path_to_database / 'poses/'


    # names = ['d110_h135_z20', 'd160_h175_z20', 'd110_h125_z2']

    # path_to_visualization = Path('/Users/eric/Documents/Studies/MSc Robotics/Thesis/Visualization/')

    # for name in names:

    #     ModelConversion.visualize_depth_map(path_to_depth, name, path_to_visualization)

    #     depth_name = name.split('.')[0]
    #     if not depth_name.endswith('_depth'):
    #         depth_name += '_depth'
    #     depth_name = depth_name + '.npz'
    #     depth_map = np.load(path_to_depth / depth_name)['depth']

    #     camera_intrinsics = [995.5555555555555, 995.5555555555555, 512.0, 512.0]

    #     pose_name = name.split('.')[0] + '.txt'
    #     pose_cam_sfm = np.loadtxt(path_to_poses / pose_name)


    #     scene_coordinates = ModelConversion.convert_depth_map_to_scene_coordinate_map(depth_map, camera_intrinsics, pose_cam_sfm)

    #     ModelConversion.visualize_scene_coordinate_map(scene_coordinates, path_to_visualization)



    
    # path_to_ground_truth = Path('/Users/eric/Documents/Studies/MSc Robotics/Thesis/Evaluation/notre_dame_B/ground_truth/')
    # path_to_depth = path_to_ground_truth / 'renders/depth/'

    # ModelConversion.visualize_depth_map(path_to_depth, 'query_00870470_3859452456_depth.npz')




    # path_to_ground_truth = Path('/Users/eric/Documents/Studies/MSc Robotics/Thesis/Evaluation/notre_dame_E/ground_truth/')
    # path_to_database = Path('/Users/eric/Documents/Studies/MSc Robotics/Thesis/Evaluation/notre_dame_E/inputs/database/')

    # model_conversion = ModelConversion(path_to_ground_truth, path_to_database)
    # model_conversion.convert_render_intrinsics_and_poses_to_colmap_format(from_blender_format=True)
    # model_conversion.convert_depth_maps_from_exr_to_npz()
