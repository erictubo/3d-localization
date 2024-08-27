import numpy as np
from pathlib import Path
from pyquaternion import Quaternion

from hloc import extract_features, pairs_from_retrieval

from interface_blender import render_query
from interface_meshloc import run_meshloc

from data import Model, CadModel
from colmap_model import ColmapModelReader, ColmapModelWriter
from model_conversion import ModelConversion
from visualization import Visualization


if __name__ == "__main__":

    # model_names = ['Reichstag']
    # cad_model_ids = ['A', 'B']

    model_names = ['Notre Dame']
    cad_model_ids = ['B']

    # Limit for GT rendering
    limit = 4000

    # Number of matches for image retrieval (MeshLoc)
    num_matched = 25


    for model_name in model_names:
        model = Model(model_name)
        
        for cad_model_id in cad_model_ids:
            cad_model = CadModel(model, cad_model_id)


            # 1. INPUT & GROUND TRUTH CONVERSION
            print('1. Input & Ground Truth Conversion...')
            print('   (Requires 3D Registration matrix T_ref)')

            colmap_model = ColmapModelReader(model.path_to_reference_model)
            query_names = colmap_model.get_all_image_names()

            # 1.1. Write query intrinsics text file (input)
            colmap_model.write_query_intrinsics_text_file(
                cad_model.path_to_query,
                query_names,
                file_name='queries.txt'
            )

            # 1.2. Write query poses text file (ground truth)
            colmap_model.write_query_poses_text_file(
                cad_model.path_to_ground_truth,
                query_names,
                file_name='cam_sfm_poses.txt'
            )

            # 1.3. Convert database intrinsics & poses from CAD to COLMAP format (input)
            cad_to_colmap = ModelConversion(cad_model.path_to_ground_truth, cad_model.path_to_database)

            print('   Converting intrinsics and poses to COLMAP format ...')
            cad_to_colmap.convert_render_intrinsics_and_poses_to_colmap_format(from_blender_format=True)

            print('   Converting depth maps to NPZ ...')
            cad_to_colmap.convert_depth_maps_from_exr_to_npz()

            print('   Creating scene coordinate maps ...')
            cad_to_colmap.convert_depth_to_scene_coordinate_maps()


            # 1.4. Convert ground truth query poses from COLMAP to CAD format for rendering
            ground_truth_poses_cam_sfm = ColmapModelWriter.read_poses_text_file(cad_model.path_to_ground_truth, 'cam_sfm_poses.txt', quaternion_first=False)

            colmap_to_cad = ModelConversion(cad_model.path_to_ground_truth)
            
            ground_truth_poses_cad_cam = colmap_to_cad.transform_poses_from_colmap_to_cad_format(ground_truth_poses_cam_sfm, to_blender_format=True)

            ColmapModelReader.write_poses_text_file(
                poses=ground_truth_poses_cad_cam,
                path_to_output=cad_model.path_to_ground_truth,
                file_name='cad_cam_poses.txt',
                quaternion_first=False
            )


            # 1.5. Render & overlay ground truth query poses
            if input("Render ground truth query poses? (y/n): ") == 'y':

                render_query(
                    blend_file = str(cad_model.blend_file),
                    target_name = cad_model.target_name,
                    render_dir = str(cad_model.path_to_ground_truth / 'renders/'),
                    intrinsics_file = str(cad_model.path_to_query / 'queries.txt'),
                    poses_file = str(cad_model.path_to_ground_truth / 'cad_cam_poses.txt'),
                    quaternion_first = False,
                    limit=limit,
                )

                Visualization.overlay_query_and_rendered_images(
                        cad_model.path_to_query_images,
                        cad_model.path_to_ground_truth / 'renders/images/',
                        cad_model.path_to_ground_truth / 'overlays/',
                        )




            # 2. IMAGE RETRIEVAL
            print('2. Image Retrieval...')

            if input("Retrieve images for MeshLoc? (y/n): ") == 'y':

                # 2.1. Extract global features
                cad_model.path_to_global_features = extract_features.main(
                    conf=extract_features.confs['netvlad'],
                    image_dir=cad_model.path_to_inputs,
                    export_dir=cad_model.path_to_features,
                )

                # 2.2 Match global features
                pairs_from_retrieval.main(
                    descriptors=cad_model.path_to_global_features,
                    output=cad_model.path_to_retrieval_pairs,
                    num_matched=num_matched,
                    db_prefix='database/images/',
                    query_prefix='query/images/',
                )

                # 2.3 Convert retrieval pairs to MeshLoc format
                with open(cad_model.path_to_retrieval_pairs, 'r') as file_in:
                    with open(cad_model.path_to_retrieval_pairs_meshloc, 'w') as file_out:
                        for line in file_in.readlines():
                            query, db = line.strip().split(' ')
                            query, db = query.split('/')[-1], db.split('/')[-1]
                            file_out.write(f'{query}, {db}, 0.0\n')




            # 3. LOCALIZATION PIPELINE
            print('3. Localization Pipeline...')
            print('   (Run MeshLoc separately)')

            print("Confirm MeshLoc run completion (press Enter):")
            input()




            # 4. EVALUATION
            print('4. Evaluation...')

            if input("Evaluate? (y/n): ") == 'y':

                output_poses_file = f'{num_matched}_patch2pix_aachen_v1_1__20.0_keypoint_clusters_POSELIB+REF_min_10000_max_100000_ref_1.0_0.25_bias_0.0_0.0.txt'
                config = 'patch2pix/'


                # 4.1. Convert to COLMAP format for visualization
                ColmapModelWriter.write_poses_text_file_to_colmap_format(cad_model.path_to_meshloc_out / config, output_poses_file, quaternion_first=True)


                # 4.2. Convert output query poses to CAD format for rendering
                output_poses_cam_sfm = ColmapModelWriter.read_poses_text_file(cad_model.path_to_meshloc_out / config, output_poses_file, quaternion_first=True)

                output_poses_cad_cam = colmap_to_cad.transform_poses_from_colmap_to_cad_format(output_poses_cam_sfm, to_blender_format=True)
                
                ColmapModelReader.write_poses_text_file(
                    poses=output_poses_cad_cam,
                    path_to_output=cad_model.path_to_meshloc_out / config,
                    file_name='cad_cam_poses.txt',
                    quaternion_first=False
                )


                # 4.3. Render output query poses & overlay with query images

                if input("Render output query poses? (y/n): ") == 'y':

                    render_query(
                        blend_file = str(cad_model.blend_file),
                        target_name = cad_model.target_name,
                        render_dir = str(cad_model.path_to_meshloc_out / config / 'renders/'),
                        intrinsics_file = str(cad_model.path_to_query / 'queries.txt'),
                        poses_file = str(cad_model.path_to_meshloc_out / config / 'cad_cam_poses.txt'),
                        quaternion_first = False,
                        limit=limit,
                    )

                    Visualization.overlay_query_and_rendered_images(
                        cad_model.path_to_query_images,
                        cad_model.path_to_meshloc_out / config / 'renders/images',
                        cad_model.path_to_meshloc_out / config / 'overlays/',
                        )


                # 4.5. Calculate metrics (in CAD frame) using ground_truth_poses_cad_cam and output_poses_cad_cam

                for query_name in output_poses_cad_cam.keys():
                    print(f'Query {query_name}:')
                    ground_truth_translation = ground_truth_poses_cad_cam[query_name][:3]
                    output_translation = output_poses_cad_cam[query_name][:3]

                    ground_truth_rotation = Quaternion(ground_truth_poses_cad_cam[query_name][3:])
                    output_rotation = Quaternion(output_poses_cad_cam[query_name][3:])

                    # Calculate translation error
                    translation_errors = output_translation - ground_truth_translation
                    print(f'Translation errors [m]: {translation_errors}')

                    # Calculate rotation error
                    rotation_error = output_rotation * ground_truth_rotation.inverse
                    rotation_error = rotation_error.angle * 180 / np.pi
                    print(f'Rotation error [deg]: {rotation_error}')
