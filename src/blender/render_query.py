import bpy
import sys
import os
import numpy as np
from math import pi
from pathlib import Path
from typing import Tuple

sys.path.append('/Users/eric/Developer/model_localization/blender')

from render import Blender
from render_data import evaluation_dir, models


def calculate_field_of_view_from_intrinsics(
        camera_model: str,
        w: int, h: int,
        camera_params: np.ndarray,
    ) -> float:
    """
    Compute field of view (FoV) in degrees.
    To match Blender:
    - Landscape format -> FoV along horizontal axis
    - Portrait format -> FoV along vertical axis
    """

    if camera_model == 'PINHOLE':
        assert camera_params.shape == (4,), camera_params.shape
        fx, fy, cx, cy = camera_params
        if w > h:
            fov = 2 * np.arctan(w / (2 * fx))
            fov_deg = fov * 180/pi
        else:
            fov = 2 * np.arctan(h / (2 * fy))
            fov_deg = fov * 180/pi
    else:
        raise ValueError(f"Camera model {camera_model} not implemented.")

    return fov_deg

def render_query_poses(
        intrinsics_file: str,
        poses_file: str,
        first: str = 't'
    ) -> None:
    """
    Render query images with pose and intrinsics.
    """
    
    with open(intrinsics_file, 'r') as f:
        intrinsics_data = f.readlines()
        for line in intrinsics_data:
            items = line.split(' ')
            query_name, camera_model, w, h, *camera_params = items
            camera_params = np.array(camera_params, dtype=float)
            w, h = int(w), int(h)

            fov_deg = calculate_field_of_view_from_intrinsics(camera_model, w, h, camera_params)

            print(f"Query: {query_name}, FoV: {fov_deg:.2f} deg")

            with open(poses_file, 'r') as g:
                poses_data = g.readlines()
                for line in poses_data:
                    items = line.split(' ')
                    pose_name = items[0]
                    if query_name == pose_name:
                        if first == 't':
                            pose = np.array(items[1:], dtype=float)
                        elif first == 'q':
                            pose = np.array(items[1:], dtype=float)
                            pose = np.concatenate([pose[4:], pose[:4]])
                        break

            print('Camera model:', camera_model)
            print('Image size:', w, h)
            print('Camera params:', camera_params)
            print('Pose:', pose)

            blender.set_camera_pose(pose)
            blender.set_camera_intrinsics(w, h, fov_deg, 'fov')

            id = f'query_{query_name.replace(".jpg", "")}'
            blender.render(id)


if __name__ == '__main__':

    for model in models:

        blend_file = models[model]['blend_file']
        target_name = models[model]['target_name']
        prefix = models[model]['prefix']

        dataset_dir = evaluation_dir + prefix

        ground_truth_dir = dataset_dir + 'ground_truth/'
        inputs_dir = dataset_dir + 'inputs/'
        output_dir = dataset_dir + 'outputs/'

        query_dir = inputs_dir + 'query/'
        query_images_dir = query_dir + 'images/'


        ground_truth_render_dir = ground_truth_dir + 'renders/'

        meshloc_out_dir = output_dir + 'meshloc_out/'
        meshloc_out_render_dir = meshloc_out_dir + 'renders/'


        intrinsics_file = query_dir + 'queries.txt'


        render_ground_truth = False
        if render_ground_truth:

            blender = Blender(
                blend_file=blend_file,
                render_dir=ground_truth_render_dir,
                target_name=target_name,
                )

            ground_truth_poses_file = ground_truth_dir + 'cad_cam_poses.txt'
            
            render_query_poses(
                intrinsics_file,
                ground_truth_poses_file,
                )
        
        render_meshloc = True
        if render_meshloc:

            blender = Blender(
                blend_file=blend_file,
                render_dir=meshloc_out_render_dir,
                target_name=target_name,
                )

            meshloc_poses_file = meshloc_out_dir + '20_superglue_aachen_v1_1__20.0_keypoint_clusters_POSELIB+REF_min_10000_max_100000_ref_1.0_0.25_bias_0.0_0.0.txt'
            
            render_query_poses(
                intrinsics_file,
                meshloc_poses_file,
                first='q',
            )


# OLD METHOD OF TRANSFERRING QUERY DATA:


# with open(path_to_ground_truth / 'cad_cam_poses.txt', 'w') as f:
    #     f.write('')

    # for query_name in query_names:
    #     query_id = sfm_data.get_image_id(query_name)

    #     query_intrinsics = sfm_data.get_intrinsics(query_id)
    #     # print('Camera intrinsics: ', query_intrinsics)

    #     # SFM pose in camera frame
    #     pose_cam_sfm = sfm_data.get_pose(query_id)
    #     # print('SFM pose (CAM frame): ', pose_cam_sfm)

    #     # Camera pose in CAD frame
    #     pose_cad_cam = cad_data.convert_query_pose_to_cad_frame(pose_cam_sfm)
    #     # print('CAM pose (CAD frame): ', pose_cad_cam)
        
    #     with open(path_to_ground_truth / 'cad_cam_poses.txt', 'a') as f:
    #         f.write(query_name + ' ' + ' '.join(map(str, pose_cad_cam)) + '\n')
    

# def read_query_ground_truth(
#         path_to_outputs: Path,
#         query_name: str,
#     ) -> Tuple[Tuple[str, np.ndarray, int, int], np.ndarray]:
#     """
#     Retrieve query data (camera intrinsics, pose) from text file.
#     """
#     with open(str(path_to_outputs) + '/' + 'query_ground_truth/' + query_name.replace('.jpg', '.txt').replace('query/',''), 'r') as f:
#         data = f.readlines()
#         for line in data:
#             items = line.split(': ')
#             key, value = items[0], items[1].strip()
#             if key == 'Camera model':
#                 camera_model = value
#             elif key == 'Image size':
#                 w, h = map(int, value.split())
#             elif key == 'Camera parameters':
#                 params = np.array(value.split(), dtype=float)
#             elif key == 'Pose':
#                 pose = np.array(value.split(), dtype=float)
    
#     assert params.shape == (4,), params.shape
#     assert pose.shape == (7,), pose.shape

#     return (camera_model, w, h, params), pose


# def render_query_poses(query_images_dir: str, intrinsics_dir: str, poses_dir: str):
#     """
#     Render query images with pose and intrinsics.
#     """
#     path_to_images = Path(query_images_dir)
    
#     query_names = [f.name for f in path_to_images]

#     for query_name in query_names:
#         (camera_model, camera_params, w, h), pose = read_query_ground_truth(path_to_ground_truth, query_name)
#         print('Camera model:', camera_model)
#         print('Image size:', w, h)
#         print('Camera params:', camera_params)
#         print('Pose:', pose)

#         fov_deg = get_field_of_view_from_intrinsics(camera_model, camera_params, w, h)

#         print(f"Query: {query_name}, FoV: {fov_deg:.2f} deg")

#         blender.set_camera_pose(pose)
#         blender.set_camera_intrinsics(w, h, fov_deg, 'fov')

#         id = f'test_{query_name.replace(".jpg", "")}'
#         blender.render(id=id)