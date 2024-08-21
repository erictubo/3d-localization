from pathlib import Path

from colmap_model import ColmapModelReader, ColmapModelWriter
from model_conversion import ModelConversion


path_to_reference = Path('/Users/eric/Documents/Studies/MSc Robotics/Thesis/3D Models/Notre Dame/Reference')
path_to_colmap_model = path_to_reference / 'dense/sparse/'
path_to_images = path_to_reference / 'dense/images/'
path_to_renders = path_to_reference / 'dense/renders/'


colmap_model_reader = ColmapModelReader(path_to_colmap_model)

image_names = colmap_model_reader.get_all_image_names()

colmap_model_reader.write_query_intrinsics_text_file(
    path_to_renders,
    image_names,
    file_name='intrinsics.txt'
    )

colmap_model_reader.write_query_poses_text_file(
    path_to_renders,
    image_names,
    file_name='poses_cam_sfm.txt'
    )

intrinsics = ColmapModelWriter.read_intrinsics_text_file(path_to_renders, 'intrinsics.txt')
poses_cam_sfm = ColmapModelWriter.read_poses_text_file(path_to_renders, 'poses_cam_sfm.txt', quaternion_first=False)

path_to_ground_truth = Path('/Users/eric/Documents/Studies/MSc Robotics/Thesis/Evaluation/notre_dame_B/ground_truth/')
model_conversion = ModelConversion(path_to_ground_truth)

poses_cad_cam = {}
for query_name, pose_cam_sfm in poses_cam_sfm.items():
    
    pose_cad_cam = model_conversion.transform_pose_from_colmap_to_cad_format(
        pose_cam_sfm=pose_cam_sfm,
        to_blender_format=True)

    poses_cad_cam[query_name] = pose_cad_cam

ColmapModelReader.write_poses_text_file(
    poses=poses_cad_cam,
    path_to_output=path_to_renders,
    file_name='poses_cad_cam.txt',
    quaternion_first=False
)