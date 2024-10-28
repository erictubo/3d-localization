import bpy
import sys
import os
import argparse
import numpy as np
from math import pi
from pathlib import Path
from typing import Tuple

script_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_directory)

from renderer import Renderer
from data import evaluation_dir

def render_query_poses(
        renderer: Renderer,
        intrinsics_file: str,
        poses_file: str,
        quaternion_first: bool,
        limit: int = None,
    ) -> None:
    """
    Render query images with pose and intrinsics.
    """
    with open(poses_file, 'r') as f, open(intrinsics_file, 'r') as g:
        poses_data = f.readlines()
        intrinsics_data = g.readlines()
        assert len(poses_data) == len(intrinsics_data)

        total = len(poses_data)
        if limit:
            total = min(limit, total)

        print(f"Rendering {total} images ...")

        for i, (pose, intrinsics) in enumerate(zip(poses_data, intrinsics_data)):
            pose_items = pose.strip().split(' ')
            name = pose_items[0]
            
            intrinsics_items = intrinsics.strip().split(' ')
            name_intrinsics = intrinsics_items[0]

            assert name == name_intrinsics, f'Name does not match.'

            if quaternion_first:
                pose = np.array(pose_items[1:], dtype=float)
                pose = np.concatenate([pose[4:], pose[:4]])
            else:
                pose = np.array(pose_items[1:], dtype=float)

            camera_model, w, h, *camera_params = intrinsics_items[1:]
            camera_params = np.array(camera_params, dtype=float)
            w, h = int(w), int(h)

            if camera_model == 'PINHOLE':
                assert camera_params.shape == (4,), camera_params.shape
                fx, fy, cx, cy = camera_params
                if w > h:
                    f = fx
                else:
                    f = fy
            else:
                raise ValueError(f"Camera model {camera_model} not implemented.")
            
            print('Query:', name)
            print('Pose:', pose)
            print('Intrinsics:', camera_model, w, h, camera_params)

            renderer.set_camera_pose(pose)
            renderer.set_camera_intrinsics(w, h, f, 'PIX', cx, cy)
            renderer.set_lighting_pose(pose)

            id = f'query_{name.replace(".jpg", "")}'
            renderer.render(id)

            print(f"Render {i} / {total} ...")

            if limit:
                if i+1 >= limit:
                    break

def main(
        blend_file: str,
        render_dir: str,
        target_name: str,
        intrinsics_file: str,
        poses_file: str,
        quaternion_first: bool,
        limit: int,
    ) -> None:

    renderer = Renderer(
        blend_file=blend_file,
        render_dir=render_dir,
        target_name=target_name,
        )

    render_query_poses(
        renderer=renderer,
        intrinsics_file=intrinsics_file,
        poses_file=poses_file,
        quaternion_first=quaternion_first,
        limit=limit,
        )


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Render query images with pose and intrinsics.')
    parser.add_argument('--blend_file', type=str, help='Path to the Blender file.')
    parser.add_argument('--target_name', type=str, help='Name of the target object.')
    parser.add_argument('--render_dir', type=str, help='Path to the render directory.')
    parser.add_argument('--intrinsics_file', type=str, help='Path to the intrinsics file.')
    parser.add_argument('--poses_file', type=str, help='Path to the poses file.')
    parser.add_argument('--quaternion_first', action='store_true', help='Whether the quaternion is first in the poses file.')
    parser.add_argument('--limit', type=int, help='Limit the number of images to render.')

    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])

    assert os.path.exists(args.blend_file), f"Blend file not found: {args.blend_file}"
    assert os.path.exists(args.intrinsics_file), f"Intrinsics file not found: {args.intrinsics_file}"
    assert os.path.exists(args.poses_file), f"Poses file not found: {args.poses_file}"

    
    main(
        blend_file=args.blend_file,
        render_dir=args.render_dir,
        target_name=args.target_name,
        intrinsics_file=args.intrinsics_file,
        poses_file=args.poses_file,
        quaternion_first=args.quaternion_first,
        limit=args.limit,
        )

    # for model in ['notre dame B']:

    #     config = 'patch2pix/'

    #     blend_file = models[model]['blend_file']
    #     target_name = models[model]['target_name']
    #     prefix = models[model]['prefix']

    #     dataset_dir = evaluation_dir + prefix

    #     ground_truth_dir = dataset_dir + 'ground_truth/'
    #     inputs_dir = dataset_dir + 'inputs/'
    #     output_dir = dataset_dir + 'outputs/'

    #     query_dir = inputs_dir + 'query/'
    #     query_images_dir = query_dir + 'images/'


    #     ground_truth_render_dir = ground_truth_dir + 'renders/'

    #     meshloc_out_dir = output_dir + 'meshloc_out/' + config
    #     meshloc_out_render_dir = meshloc_out_dir + 'renders/'


    #     intrinsics_file = query_dir + 'queries.txt'


    #     render_ground_truth = False
    #     if render_ground_truth:

    #         renderer = Renderer(
    #             blend_file=blend_file,
    #             render_dir=ground_truth_render_dir,
    #             target_name=target_name,
    #             )

    #         ground_truth_poses_file = ground_truth_dir + 'cad_cam_poses.txt'
            
    #         render_query_poses(
    #             intrinsics_file,
    #             ground_truth_poses_file,
    #             quaternion_first=False,
    #             )
        
    #     render_meshloc = True
    #     if render_meshloc:

    #         renderer = Renderer(
    #             blend_file=blend_file,
    #             render_dir=meshloc_out_render_dir,
    #             target_name=target_name,
    #             )

    #         meshloc_poses_file = meshloc_out_dir + 'cad_cam_poses.txt'
            
    #         render_query_poses(
    #             intrinsics_file,
    #             meshloc_poses_file,
    #             quaternion_first=True,
    #         )
        
    #     render_sfm = False
    #     if render_sfm:
                
    #             sfm_render_dir = '/Users/eric/Documents/Studies/MSc Robotics/Thesis/3D Models/Notre Dame/Reference/dense/renders/'
    #             sfm_intrinsics_file = sfm_render_dir + 'intrinsics.txt'
    #             sfm_poses_file = sfm_render_dir + 'poses_cad_cam.txt'

    #             renderer = Renderer(
    #                 blend_file=blend_file,
    #                 render_dir=sfm_render_dir,
    #                 target_name=target_name,
    #                 )
                    
    #             render_query_poses(
    #                 sfm_intrinsics_file,
    #                 sfm_poses_file,
    #                 quaternion_first=False,
    #             )