import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Union
# from pprint import pformat
# from types import SimpleNamespace
# import pycolmap

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from features import Features
from sfm_data import SfmData
from cad_data import CadData
from evaluation import overlay_query_and_rendered_images


if __name__ == '__main__':

    path_to_evaluation = Path('/Users/eric/Downloads/evaluation/')

    models: list = [
        'notre dame B',
        'notre dame E'
        ]

    for model in models:

        path_to_dataset = path_to_evaluation / f'{model}/'
        path_to_ground_truth = path_to_dataset / 'ground_truth/'
        path_to_inputs = path_to_dataset / 'inputs/'
        path_to_outputs = path_to_dataset / 'outputs/'

        assert path_to_ground_truth.exists(), f"Ground truth directory not found: {path_to_ground_truth}"
        assert path_to_inputs.exists(), f"Inputs directory not found: {path_to_inputs}"
        if not path_to_outputs.exists(): path_to_outputs.mkdir()

        path_to_query = path_to_inputs / 'query/'
        path_to_query_images = path_to_query / 'images/'

        path_to_database = path_to_inputs / 'database/'

        query_names = [f.name for f in path_to_query_images.glob('*')]


        # 1. Feature Extraction & Matching
        print('Image retrieval...')

        path_to_features = path_to_outputs / 'features/'
        path_to_retrieval_pairs = path_to_features / "pairs-from-retrieval.txt"

        features = Features(
            path_to_inputs,
            path_to_features,
            path_to_retrieval_pairs,
            global_feature_conf_name='netvlad',
            global_num_matched=10,
            local_feature_conf_name='superpoint_aachen',
            local_match_conf_name='superglue',
            )

        features.image_retrieval()

        # features.local_feature_matching()
        # for query_name in query_names:
        #     features.visualize_local_matches(query_name, db_limit=3, min_score=0.6)
        



        # 2. SFM vs. CAD Data Evaluation
        print('SfM vs. CAD data evaluation...')

        path_to_sfm_data = Path('/Users/eric/Downloads/notre_dame_front_facade/dense/sparse/')
        sfm_data = SfmData(path_to_sfm_data)
        sfm_data.write_query_intrinsics_text_file(path_to_inputs, query_names)
        sfm_data.write_query_poses_text_file(path_to_ground_truth, query_names)

        cad_data = CadData(path_to_ground_truth, path_to_database)

        with open(path_to_ground_truth / 'cad_cam_poses.txt', 'w') as f:
            f.write('')

        for query_name in query_names:
            query_id = sfm_data.get_image_id(query_name)

            query_intrinsics = sfm_data.get_intrinsics(query_id)
            # print('Camera intrinsics: ', query_intrinsics)

            # SFM pose in camera frame
            pose_cam_sfm = sfm_data.get_pose(query_id)
            # print('SFM pose (CAM frame): ', pose_cam_sfm)

            # Camera pose in CAD frame
            pose_cad_cam = cad_data.convert_query_pose_to_cad_frame(pose_cam_sfm)
            # print('CAM pose (CAD frame): ', pose_cad_cam)
            
            with open(path_to_ground_truth / 'cad_cam_poses.txt', 'a') as f:
                f.write(query_name + ' ' + ' '.join(map(str, pose_cad_cam)) + '\n')


        # TODO: put in separate loop or move to another file (evaluation.py)

        overlay_query_and_rendered_images(
            path_to_query_images,
            path_to_ground_truth / 'renders/images/',
            path_to_ground_truth / 'overlays/'
            )




        # 3. Localization (MeshLoc)
        print('MeshLoc localization...')
        
        # 3.1 Depth data
        # TODO: make my data compatible with MeshLoc


        # 3.2 CAD pose data
        # TODO: make my data compatible with MeshLoc

        # pose_db_cad = get_db_pose('database/image_0040.png')
        # T_db_cad = pose_to_matrix(pose_db)



# TODO: separate tasks into different files
# A. Preprocessing (global feature extraction, SFM data extraction, CAD data extraction, etc.)
# B. MeshLoc (local feature matching, localization, etc.)
# C. Evaluation (ground truth comparison, visualization, error analysis, etc.)