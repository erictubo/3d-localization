import sys
import subprocess


conda_env_name = 'immatch'
working_directory = '/Users/eric/Developer/image-matching-toolbox/'


def run_meshloc(
        database_dir: str,
        query_dir: str,
        output_dir: str,
        config: str = 'patch2pix',
        top_k: int = 25,
    ):

    # Remove trailing slashes
    if database_dir[-1] == '/': database_dir = database_dir[:-1]
    if query_dir[-1] == '/': query_dir = query_dir[:-1]
    if output_dir[-1] == '/': output_dir = output_dir[:-1]

    # Build command to run in the specified conda environment
    command = [
        'conda', 'run', '-n', conda_env_name, 'python', '../meshloc_release/localize.py',
        '--db_image_dir', f'{database_dir}/imagess',
        '--db_depth_image_dir', f'{database_dir}/depth',
        '--colmap_model_dir', database_dir,
        '--query_dir', f'{query_dir}/images',
        '--query_list', f'{query_dir}/queries.txt',
        '--match_prefix', f'{output_dir}/meshloc_match/{config}/',
        '--out_prefix', f'{output_dir}/meshloc_out/{config}/',
        '--method_name', config,
        '--method_config', 'aachen_v1.1',
        '--method_string', f'{config}_aachen_v1_1_',
        '--retrieval_pairs', f'{output_dir}/features/pairs-from-retrieval-meshloc.txt',
        '--top_k', str(top_k),
        '--max_side_length', '-1',
        '--ransac_type', 'POSELIB+REF',
        '--min_ransac_iterations', '10000',
        '--max_ransac_iterations', '100000',
        '--reproj_error', '20.0',
        '--use_orig_db_images',
        '--cluster_keypoints'
    ]

    # Execute command
    result = subprocess.run(command, cwd=working_directory, check=True, capture_output=True, text=True)
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)




    # process = subprocess.Popen(command, cwd=working_directory, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # while process.poll() is None:
    #     output = process.stdout.readline()
    #     print(output)


    # process = await asyncio.create_subprocess_exec(
    #     *command,
    #     cwd=working_directory,
    #     stdout=subprocess.PIPE,
    #     stderr=subprocess.PIPE,
    # )

    # async for line in process.stdout:
    #     print(line.decode().strip())

    # async for line in process.stderr:
    #     print(line.decode().strip())
    
    # await process.wait()




    # # Initialize conda
    # subprocess.run("conda init", shell=True, check=True)
    
    # # Activate conda environment
    # subprocess.run(f"conda activate {conda_env_name}", shell=True, check=True)
    
    # # Change to working directory
    # subprocess.run(f"cd {working_directory}", shell=True, check=True)

    # command = f"""
    #     python src/meshloc_release/localize.py \
        
    #     --db_image_dir "{database_dir}/images" \
    #     --db_depth_image_dir "{database_dir}/depth" \
    #     --colmap_model_dir "{database_dir}" \
        
    #     --query_dir "{query_dir}/images" \
    #     --query_list "{query_dir}/queries.txt" \
        
    #     --match_prefix "{output_dir}/meshloc_match/{config}/" \
    #     --out_prefix "{output_dir}/meshloc_out/{config}/" \
        
    #     --method_name {config} \
    #     --method_config aachen_v1.1 \
    #     --method_string {config}_aachen_v1_1_ \
    #     --retrieval_pairs "{output_dir}/features/pairs-from-retrieval-meshloc.txt" \
    #     --top_k {top_k} \
    #     --max_side_length -1 \
    #     --ransac_type POSELIB+REF \
    #     --min_ransac_iterations 10000 \
    #     --max_ransac_iterations 100000 \
    #     --reproj_error 20.0 \
    #     --use_orig_db_images \
    #     --cluster_keypoints
    # """

    # subprocess.run(command, shell=True)


if __name__ == '__main__':

    database_dir = '/Users/eric/Documents/Studies/MSc Robotics/Thesis/Evaluation/notre_dame_B/inputs/database'
    query_dir = '/Users/eric/Documents/Studies/MSc Robotics/Thesis/Evaluation/notre_dame_B/inputs/query'
    output_dir = '/Users/eric/Documents/Studies/MSc Robotics/Thesis/Evaluation/notre_dame_B/outputs/'

    run_meshloc(database_dir, query_dir, output_dir)