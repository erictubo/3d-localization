import numpy as np
from scipy.spatial.transform import Rotation
from pyquaternion import Quaternion
import pycolmap
from pathlib import Path

from typing import Dict, Tuple, List, Optional, Union
from pprint import pformat
from types import SimpleNamespace

from features import Features
from colmap_data import ColmapData
from transformation import pose_to_matrix, matrix_to_pose, invert_pose, invert_matrix
from transformations import decompose_matrix, compose_matrix

# import os
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def get_db_pose(path_to_images: Path, db_name: str) -> np.ndarray:
    """
    Get pose of database image from render output text file.
    Format: scalar-first (px, py, pz, qw, qx, qy, qz)
    """
    pose_name = db_name.replace('.png', '.txt').replace('image', 'pose')
    pose = np.loadtxt(path_to_images / pose_name)
    assert pose.shape == (7,), pose.shape

    return pose

def get_transformation_matrix(path: Path, name: str) -> np.ndarray:
    """
    Get reference transformation from text file.
    """
    T = np.loadtxt(path / name)
    assert T.shape == (4,4), T.shape

    return T

def convert_query_pose_to_cad_frame(pose_cam_sfm: np.ndarray, T_cad_sfm: np.ndarray) -> np.ndarray:
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
    T_cad_cam = T_cad_sfm @ T_sfm_cam
    scale, shear, angles, translate, perspective = decompose_matrix(T_cad_cam)
    T_cad_cam = compose_matrix(None, None, angles, translate)
    pose_cad_cam = matrix_to_pose(T_cad_cam)
    # print('CAM pose (CAD frame): ', pose_cad_cam)

    return pose_cad_cam


def save_query_data(path_to_outputs: Path, query_name: str, intrinsics: Tuple[str, np.ndarray, int, int], pose: np.ndarray):
    """
    Save query data (intrinsics, pose) to file.
    """
    if not path_to_outputs.exists():
        path_to_outputs.mkdir(parents=True)

    camera_model, params, w, h = intrinsics

    with open(path_to_outputs / query_name.replace('.jpg', '.txt').replace('query/',''), 'w') as f:
        f.write(camera_model + '\n')
        f.write(' '.join(map(str, params)) + '\n')
        f.write(str(w) + ' ' + str(h) + '\n')
        f.write(' '.join(map(str, pose)) + '\n')

def overlay_query_and_rendered_images(path_to_renders: Path, path_to_outputs: Path):
    """
    
    """
    render_names = [f.name for f in path_to_renders.glob('image_test_*.png')]

    print('Query names: ', query_names)

    print('Path to renders: ', path_to_renders)
    print('Render names: ', render_names)

    for query_name in query_names:

        from PIL import Image

        render_name = next((name for name in render_names if query_name.replace('.jpg','') in name), None)
        
        if render_name:
            # Load the two images
            query_image = Image.open(path_to_images / 'query/' / query_name)
            render_image = Image.open(path_to_renders / render_name)

            # Resize the images while maintaining aspect ratio
            # check that query image and render image have the same dimensions
            if query_image.size != render_image.size:
                width, height = 600, 600  # Desired size
            else:
                width, height = query_image.size
            query_image = query_image.resize((width, height), Image.Resampling.LANCZOS)
            render_image = render_image.resize((width, height), Image.Resampling.LANCZOS)

            # Create a new blank image with combined dimensions
            combined_width = width
            combined_height = height
            combined_image = Image.new('RGB', (combined_width, combined_height))

            # Paste the cropped images onto the combined image
            combined_image.paste(query_image.crop((0, 0, width//2, height//2)), (0, 0))  # Top left quarter
            combined_image.paste(query_image.crop((width//2, height//2, width, height)), (width//2, height//2))  # Bottom right quarter
            combined_image.paste(render_image.crop((0, height//2, width//2, height)), (0, height//2))  # Top right quarter
            combined_image.paste(render_image.crop((width//2, 0, width, height//2)), (width//2, 0))  # Bottom left quarter

            # Save the combined image
            path_to_overlays = path_to_outputs / 'overlays/'
            if not path_to_overlays.exists():
                 path_to_overlays.mkdir()
            combined_image.save(path_to_overlays / query_name)            


if __name__ == '__main__':

    # COLMAP SFM Model
    path_to_model = Path('/Users/eric/Downloads/notre_dame_front_facade')
    colmap_data = ColmapData(path_to_model)

    path_to_evaluation = Path('/Users/eric/Downloads/evaluation/')

    models: list = [
        # 'notre dame B',
        'notre dame E'
        ]

    for model in models:

        path_to_dataset = path_to_evaluation / f'{model}/'
        path_to_images = path_to_dataset / 'inputs/'
        path_to_outputs = path_to_dataset / 'outputs/'

        assert path_to_images.exists(), path_to_images
        if not path_to_outputs.exists():
            path_to_outputs.mkdir(parents=True)

        path_to_retrieval_pairs = path_to_outputs / "pairs-from-retrieval.txt"

        features = Features(
            path_to_images,
            path_to_outputs,
            path_to_retrieval_pairs,
            global_feature_conf_name='netvlad',
            global_num_matched=10,
            local_feature_conf_name='superpoint_aachen',
            local_match_conf_name='superglue',
            )

        features.image_retrieval()
        features.local_feature_matching()

        query_names = [f.name for f in path_to_images.glob('query/*.jpg')]

        for query_name in query_names:
            features.visualize_local_matches(query_name, db_limit=1, min_score=0.6)


        # OPTION 1 - CADLOC T_REF
        # CAD in SFM frame
        # T_sfm_cad = get_reference_transformation(path_to_images, 'reference_transformation.txt')
        # scale, shear, angles, translate, perspective = decompose_matrix(T_sfm_cad)
        # T_sfm_cad = compose_matrix(scale, None, angles, translate)

        # OPTION 2 - CLOUDCOMPARE REGISTRATION  
        # T_cad_sfm = np.array([
        #     [-5.636510, 3.172812, 9.269413, -123.011314],
        #     [-9.796731, -1.701635, -5.374708, 51.065014],
        #     [-0.113225, -10.714321, 3.598537, 6.478726],
        #     [0, 0, 0, 1],
        #     ])




        T_cad_sfm = get_transformation_matrix(path_to_images, 'T_cad_sfm.txt')
        T_sfm_cad = np.linalg.inv(T_cad_sfm)
        
        scale, shear, angles, translate, perspective = decompose_matrix(T_cad_sfm)
        q_cad_sfm = Quaternion(matrix=Rotation.from_euler('xyz', angles).as_matrix()).inverse
        pose_cad_sfm = np.hstack([translate, q_cad_sfm.q])
        s_cad_sfm = sum(scale) / len(scale)
        print('SFM pose (CAD frame): ', pose_cad_sfm)
        print('SFM vs. CAD scale: ', s_cad_sfm)


        for query_name in query_names:
            query_id = colmap_data.get_image_id(query_name)

            query_intrinsics = colmap_data.get_intrinsics(query_id)
            print('Camera intrinsics: ', query_intrinsics)

            pose_cam_sfm = colmap_data.get_pose(query_id)
            print('SFM pose (CAM frame): ', pose_cam_sfm)

            pose_cad_cam = convert_query_pose_to_cad_frame(pose_cam_sfm, T_cad_sfm)
            print('CAM pose (CAD frame): ', pose_cad_cam)

            save_query_data(path_to_outputs / 'query_data/', query_name, query_intrinsics, pose_cad_cam)


        path_to_renders = Path(f'/Users/eric/Library/Mobile Documents/com~apple~CloudDocs/Blender/renders/{model}')

        overlay_query_and_rendered_images(path_to_renders, path_to_outputs)


        # TODO: Localization
        # make MeshLoc compatible with my data (depth npy/npz files, etc.)
        # Insert localization pipeline here



        # pose_db_cad = get_db_pose('database/image_0040.png')
        # T_db_cad = pose_to_matrix(pose_db)


