import numpy as np
from pathlib import Path
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation
from transformations import decompose_matrix, compose_matrix
from typing import Tuple, Dict, List
import collections
from matplotlib import pyplot as plt
import OpenEXR
import Imath

from src.model_conversion import convert_pose_to_matrix, convert_matrix_to_pose
from hloc.utils.read_write_model import write_cameras_binary, write_images_binary, write_points3D_binary, CAMERA_MODEL_NAMES


class CadData:

    def __init__(
            self,
            path_to_ground_truth: Path,
            path_to_database: Path,
            images_prefix: str = 'images/',
            poses_prefix: str = 'poses/',
            depth_prefix: str = 'depth/',
            ):
        
        self.path_to_ground_truth = path_to_ground_truth
        self.path_to_database = path_to_database

        self.path_to_poses = path_to_database / poses_prefix
        self.path_to_images = path_to_database / images_prefix
        self.path_to_depth = path_to_database / depth_prefix

        self.image_names = [f.name for f in self.path_to_images.iterdir() if f.is_file() and not f.name.startswith('.')]
        self.image_names = [name for name in self.image_names if name.split('.')[-1] in ['png', 'jpg', 'jpeg']]
        
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
    
    def read_database_pose(self, image_name: str) -> np.ndarray:
        """
        Get pose of database image from render output text file.
        Format: scalar-first (px, py, pz, qw, qx, qy, qz)
        """
        name = image_name.split('.')[0]
        pose_name = name + '.txt'
        pose = np.loadtxt(self.path_to_poses / pose_name)
        assert pose.shape == (7,), pose.shape

        return pose

    def convert_query_pose_to_cad_frame(self, pose_cam_sfm: np.ndarray) -> np.ndarray:
        """
        Get pose of query image in CAD frame.
        """
        # SFM in CAM frame
        # rotate 180 degrees about the x-axis to match Blender camera (facing -z direction)
        T_cam_sfm = compose_matrix(None, None, [np.pi, 0, 0], [0, 0, 0]) @ convert_pose_to_matrix(pose_cam_sfm)

        # CAM in SFM frame
        T_sfm_cam = np.linalg.inv(T_cam_sfm)
        # pose_sfm_cam = convert_matrix_to_pose(T_sfm_cam)
        # print('CAM pose (SFM frame): ', pose_sfm_cam)

        # CAM in CAD frame
        T_cad_cam = self.T_cad_sfm @ T_sfm_cam
        scale, shear, angles, translate, perspective = decompose_matrix(T_cad_cam)
        T_cad_cam = compose_matrix(None, None, angles, translate)
        pose_cad_cam = convert_matrix_to_pose(T_cad_cam)
        # print('CAM pose (CAD frame): ', pose_cad_cam)

        return pose_cad_cam
    
    def reverse_cad_pose_to_colmap_format(self, pose_cad_cam: np.ndarray) -> np.ndarray:
        """
        Reverse render pose in CAD frame to COLMAP format (pose_cam_cad).
        """
        T_cad_cam = convert_pose_to_matrix(pose_cad_cam)
        T_cam_cad = compose_matrix(None, None, [np.pi, 0, 0], [0, 0, 0]) @ np.linalg.inv(T_cad_cam)
        pose_cam_cad = convert_matrix_to_pose(T_cam_cad)

        return pose_cam_cad

    def convert_render_data_to_colmap_format(
            self,
            model_name: str = None,
            ):
        # - images[image_id] = Image(id, qvec, tvec, camera_id, name, xys, point3D_ids)   
        # - cameras[camera_id] = Camera(id, model, width, height, params)
        # - points3D[point3D_id] = Point3D(id, xyz, rgb, error, image_ids, point2D_idxs)

        Camera = collections.namedtuple(
            "Camera", ["id", "model", "width", "height", "params"]
        )
        Image = collections.namedtuple(
            "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"]
        )
        Point3D = collections.namedtuple(
            "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"]
        )

        # read intrinsics.txt file in self.path_to_database with w, h, f_mm

        with open(self.path_to_database / 'intrinsics.txt', 'r') as file:
            lines = file.readlines()
            w, h, f_mm = [float(x) for x in lines[0].split()]
            w, h = int(w), int(h)

        print('w, h, f_mm: ', w, h, f_mm)

        # w, h = image_size[0], image_size[1]
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

        images = {}

        for i, image_name in enumerate(self.image_names):

            # name = image_name.split('.')[0]
            image_id = i+1

            pose_cad_cam = self.read_database_pose(image_name)
            pose_cam_cad = self.reverse_cad_pose_to_colmap_format(pose_cad_cam)
            tvec = pose_cam_cad[:3]
            qvec = pose_cam_cad[3:]

            xys = np.array([])
            point3D_ids = np.array([])

            images[image_id] = Image(
                id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera.id,
                name=image_name,
                xys=xys,
                point3D_ids=point3D_ids,
            )
        
        write_images_binary(images, self.path_to_database / 'images.bin')

        points3d = {}
        # Read bounding box coordinates from database
        with open(self.path_to_database / 'bounding_box.txt', 'r') as file:
            lines = file.readlines()
            for i, line in enumerate(lines):
                line.strip()
                xyz = np.array([float(x) for x in line.split()])

                points3d[i+1] = Point3D(
                    id=i+1,
                    xyz=xyz,
                    rgb=np.array([0, 0, 0]),
                    error=0,
                    image_ids=np.array([]),
                    point2D_idxs=np.array([]),
                )

        write_points3D_binary(points3d, self.path_to_database / 'points3D.bin')

        # points3d = {}

        # if model_name == 'notre dame B':
        #     xyzs = np.array([
        #         [6.0942, -4.1046, 45.9717], # center
        #         [-68.25932312011719, -60.856727600097656, -1.1421051025390625],
        #         [-68.25932312011719, -60.856727600097656, 93.08549499511719],
        #         [-68.25932312011719, 52.647544860839844, -1.1421051025390625],
        #         [-68.25932312011719, 52.647544860839844, 93.08549499511719],
        #         [80.44776916503906, -60.856727600097656, -1.1421051025390625],
        #         [80.44776916503906, -60.856727600097656, 93.08549499511719],
        #         [80.44776916503906, 52.647544860839844, -1.1421051025390625],
        #         [80.44776916503906, 52.647544860839844, 93.08549499511719],
        #     ])
        # elif model_name == 'notre dame E':
        #     xyzs = np.array([
        #         [0.0000, 79.4676, 52.5500], # center
        #         [-35.53199005126953, -3.115997314453125, 0.0],
        #         [-35.53199005126953, -3.115997314453125, 105.10000610351562],
        #         [-35.53199005126953, 162.0512237548828, 0.0],
        #         [-35.53199005126953, 162.0512237548828, 105.10000610351562],
        #         [35.53199005126953, -3.115997314453125, 0.0],
        #         [35.53199005126953, -3.115997314453125, 105.10000610351562],
        #         [35.53199005126953, 162.0512237548828, 0.0],
        #         [35.53199005126953, 162.0512237548828, 105.10000610351562],
        #     ])
        # elif model_name == None:
        #     xyzs = np.array([[0, 0, 0]])
        # else:
        #     raise ValueError(f"Invalid model name: {model_name}")

        # for i in range(len(xyzs)):
        #     points3d[i] = Point3D(
        #         id=i,
        #         xyz=xyzs[i],
        #         rgb=np.array([0, 0, 0]),
        #         error=0,
        #         image_ids=np.array([]),
        #         point2D_idxs=np.array([]),
        #     )

        # write_points3D_binary(points3d, self.path_to_database / 'points3D.bin')

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


if __name__ == '__main__':

    path_to_depth = Path('/Users/eric/Downloads/evaluation/notre_dame_B/inputs/database/depth/')

    names = ['0001_depth', '0002_depth', '0003_depth']

    for name in names:

        CadData.visualize_depth_map(path_to_depth, name)