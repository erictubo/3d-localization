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

from data import path_to_evaluation, reference_models

if __name__ == '__main__':

    for reference_model in reference_models:

        path_to_sfm_data = Path(reference_model['sfm_model'])
        sfm_data = SfmData(path_to_sfm_data)

        for cad_model in reference_model['cad_models']:

            prefix = reference_model['cad_models'][cad_model]['prefix']

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
            path_to_retrieval_pairs = path_to_features / "pairs-from-retrieval.txt"

            query_names = [f.name for f in path_to_query_images.glob('*')]
            print('Query images: ', query_names)



            # 1. Image Retrieval
            print('1. Image Retrieval...')
            print('   (Requires Database Rendering - run render_database.py using Blender task)')

            features = Features(
                path_to_inputs,
                path_to_features,
                path_to_retrieval_pairs,
                global_feature_conf_name='netvlad',
                global_num_matched=20,
                local_feature_conf_name='superpoint_aachen',
                local_match_conf_name='superglue',
                )

            features.image_retrieval()
            features.write_meshloc_retrieval_pairs()


            # Local Feature Matching done by Localization Pipeline
            # features.local_feature_matching()
            # for query_name in query_names:
            #     features.visualize_local_matches(query_name, db_limit=3, min_score=0.6)
            


            # 2. Ground Truth Conversion
            print('2. Ground Truth Conversion...')
            print('   (Requires 3D Registration - run manually using CloudCompare)')

            sfm_data.write_query_intrinsics_text_file(path_to_query, query_names)
            sfm_data.write_query_poses_text_file(path_to_ground_truth, query_names)

            cad_data = CadData(path_to_ground_truth, path_to_database)
            cad_data.convert_render_data_to_colmap_format()
            cad_data.convert_depth_maps_from_exr_to_npz()



            # 3. Localization Pipeline
            print('Localization Pipeline...')
            print('   (For MeshLoc, run terminal command in image-matching-toolbox using immatch conda environment)')
            
            


            # 4. Evaluation
            print('4. Evaluation...')
            print('   (Requires Query Rendering - run render_query.py using Blender task)')

            overlay_query_and_rendered_images(
                path_to_query_images,
                path_to_ground_truth / 'renders/images/',
                path_to_ground_truth / 'overlays/'
                )

            overlay_query_and_rendered_images(
                path_to_query_images,
                path_to_outputs / 'meshloc_out/renders/images',
                path_to_outputs / 'meshloc_out/overlays/'
                )
