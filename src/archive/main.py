import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Union
# from pprint import pformat
# from types import SimpleNamespace
# import pycolmap

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from data import path_to_evaluation, models
from features import Features
from colmap_model import ColmapModelReader, ColmapModelWriter, ColmapModelConverter
from model_conversion import ModelConversion
from visualization import Visualization

if __name__ == '__main__':

    for model in models:

        path_to_sfm_model = model['sfm model']
        sfm_model = ColmapModelReader(path_to_sfm_model)

        for cad_model in ['notre dame B']:

            prefix = model['cad models'][cad_model]['prefix']

            path_to_dataset = path_to_evaluation / f'{prefix}'
            path_to_ground_truth = path_to_dataset / 'ground_truth/'
            path_to_inputs = path_to_dataset / 'inputs/'
            path_to_outputs = path_to_dataset / 'outputs/'

            assert path_to_ground_truth.exists(), f"Ground truth directory not found: {path_to_ground_truth}"
            assert path_to_inputs.exists(), f"Inputs directory not found: {path_to_inputs}"
            if not path_to_outputs.exists(): path_to_outputs.mkdir()

            path_to_query = path_to_inputs / 'query/'
            path_to_query_images = path_to_query / 'images/'
            path_to_database = path_to_inputs / 'database/'

            path_to_features = path_to_outputs / 'features/'
            path_to_meshloc_out = path_to_outputs / 'meshloc_out/'
            path_to_retrieval_pairs = path_to_features / "pairs-from-retrieval.txt"          

            query_names = [f.name for f in path_to_query_images.iterdir() if (f.is_file() and not f.name.startswith('.'))]
            query_names = [name for name in query_names if name.split('.')[-1] in ['png', 'jpg', 'jpeg']]
            print('Query images: ', query_names)



            # 1. Image Retrieval
            print('1. Image Retrieval...')
            print('   (Requires Database Rendering - run render_database.py using Blender task)')

            features = Features(
                path_to_inputs,
                path_to_features,
                path_to_retrieval_pairs,
                global_feature_conf_name='netvlad',
                global_num_matched=25,
                local_feature_conf_name='superpoint_aachen',
                local_match_conf_name='superglue',
            )

            features.retrieve_image_pairs()

            # Local Feature Matching done by Localization Pipeline
            # features.match_local_features()
            # for query_name in query_names:
            #     features.visualize_local_matches(query_name, db_limit=3, min_score=0.6)



            # 2. Ground Truth Conversion
            print('2. Ground Truth Conversion...')
            print('   (Requires 3D Registration - run manually using CloudCompare)')

            # 2.1 In COLMAP format for visualization
            sfm_model.write_query_intrinsics_text_file(path_to_query, query_names)
            sfm_model.write_query_poses_text_file(path_to_ground_truth, query_names)

            model_conversion = ModelConversion(path_to_ground_truth, path_to_database)
            model_conversion.convert_depth_maps_from_exr_to_npz()
            model_conversion.convert_render_intrinsics_and_poses_to_colmap_format(
                from_blender_format=True)


            # 2.2 In CAD format for rendering

            sfm2cad_conversion = ModelConversion(path_to_ground_truth)

            poses_cam_sfm = ColmapModelWriter.read_poses_text_file(path_to_ground_truth, 'cam_sfm_poses.txt', quaternion_first=False)

            poses_cad_cam = {}
            for query_name, pose_cam_sfm in poses_cam_sfm.items():
                
                pose_cad_cam = sfm2cad_conversion.transform_pose_from_colmap_to_cad_format(
                    pose_cam_sfm=pose_cam_sfm,
                    to_blender_format=True)

                poses_cad_cam[query_name] = pose_cad_cam

            ColmapModelReader.write_poses_text_file(
                poses=poses_cad_cam,
                path_to_output=path_to_ground_truth,
                file_name='cad_cam_poses.txt',
                quaternion_first=False
            )




            # 3. Localization Pipeline
            print('Localization Pipeline...')
            print('   (For MeshLoc, run terminal command in image-matching-toolbox using immatch conda environment)')
            
            # TODO: read MeshLoc output poses
            meshloc_poses_file = '25_superglue_aachen_v1_1__20.0_keypoint_clusters_POSELIB+REF_min_10000_max_100000_ref_1.0_0.25_bias_0.0_0.0.txt'

            # TODO: convert back from colmap format to blender format for rendering
            with open(path_to_meshloc_out / meshloc_poses_file, 'r') as file_in:
                with open(path_to_meshloc_out / 'cad_cam_poses.txt', 'w') as file_out:

                    poses_data = file_in.readlines()
                    for line in poses_data:
                        line.strip()
                        items = line.split(' ')
                        query_name = items[0]
                        pose_cam_sfm = np.array([float(i) for i in items[1:]])
                        pose_cad_cam = model_conversion.convert_query_pose_to_cad_frame(pose_cam_sfm)
                        file_out.write(query_name + ' ' + ' '.join(map(str, pose_cad_cam)) + '\n')
                    
            # TODO: convert poses to COLMAP format for visualization



            # 4. Evaluation
            print('4. Evaluation...')
            print('   (Requires Query Rendering - run render_query.py using Blender task)')





            Visualization.overlay_query_and_rendered_images(
                path_to_query_images,
                path_to_ground_truth / 'renders/images/',
                path_to_ground_truth / 'overlays/'
                )

            Visualization.overlay_query_and_rendered_images(
                path_to_query_images,
                path_to_outputs / 'meshloc_out/renders/images',
                path_to_outputs / 'meshloc_out/overlays/'
                )
