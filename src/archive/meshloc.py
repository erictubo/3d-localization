import sys
import os

sys.path.append('/Users/eric/Developer/cad_localization/pipelines/meshloc')
sys.path.append('/Users/eric/Developer/meshloc_release')
sys.path.append('/Users/eric/Developer')


from ....meshloc_release import localize

def run_meshloc(
        # colmap_model_dir,
        # db_images_dir,
        # db_depth_images_dir,
        # query_list_file,
        # query_images_dir,
        # retrieval_pairs_file,
        # meshloc_out_dir,
        # meshloc_match_dir,
        # method_name: str = 'superglue',
        # method_config: str = 'aachen_v1.1',
        # method_string: str = 'superglue_aachen_v1_1',
        # top_k: int = 20,
        # ransac_type: str = 'POSELIB+REF',
):
    # Simulate command line arguments
    sys.argv = [
        '',  # Script name, not necessary here
        '--colmap_model_dir', colmap_model_dir,
        '--db_image_dir', db_images_dir,
        '--db_depth_image_dir', db_depth_images_dir,
        '--query_list', query_list_file,
        '--query_dir', query_images_dir,
        '--retrieval_pairs', retrieval_pairs_file,
        '--method_name', method_name,
        '--method_config', method_config,
        '--method_string', method_string,
        '--out_prefix', meshloc_out_dir,
        '--match_prefix', meshloc_match_dir,
        '--top_k', top_k,
        '--reproj_error', '20.0',
        '--use_orig_db_images',
        # '--triangulate',
        # '--merge_3D_points',
        '--cluster_keypoints',
        # '--covisibility_filtering',
        # '--all_matches_ransac',
        '--min_ransac_iterations', '1000',
        '--max_ransac_iterations', '100000',
        '--max_side_length', '-1',
        '--ransac_type', ransac_type,
        # '--rendering_postfix', '.png',
        # '--refinement_range', '1.0',
        # '--refinement_step', '0.25',
        # '--bias_x', '0.0',
        # '--bias_y', '0.0'
    ]

    localize.main()

if __name__ == "__main__":


    method_name = 'superglue'
    method_config = 'aachen_v1.1'
    method_string = 'superglue_aachen_v1_1'
    top_k = 20
    ransac_type = 'POSELIB+REF'

    models = ['notre_dame_B', 'notre_dame_E']

    for model in models:
        
        evaluation_dir = '/Users/eric/Downloads/evaluation/'
        data_dir = evaluation_dir + model + '/'

        inputs_dir = data_dir + 'inputs/'
        outputs_dir = data_dir + 'outputs/'

        db_images_dir = inputs_dir + 'database/images/'
        colmap_model_dir = inputs_dir + 'database/'
        db_depth_images_dir = inputs_dir + 'database/depth/'

        query_images_dir = inputs_dir + 'query/images/'
        query_list_file = query_images_dir + 'queries.txt'
        retrieval_pairs_file = outputs_dir + 'features/' + 'pairs-from-retrieval-meshloc.txt'

        meshloc_out_dir = outputs_dir + 'meshloc_out/'
        meshloc_match_dir = outputs_dir + 'meshloc_match/'

        run_meshloc()