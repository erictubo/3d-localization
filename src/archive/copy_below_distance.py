import numpy as np

from data import path_to_data

path_to_input = path_to_data / 'GLACE/notre_dame'
path_to_output = path_to_data / 'GLACE/notre_dame_closer'

if not path_to_output.exists(): path_to_output.mkdir()

i = 0

for split in ['train', 'test']:
    path_to_input_split = path_to_input / split
    path_to_output_split = path_to_output / split

    if not path_to_output_split.exists(): path_to_output_split.mkdir()

    subpaths = ['rgb', 'calibration', 'poses', 'init'] # , 'depth']

    for subpath in subpaths:
        path = path_to_output_split / subpath
        if not path.exists(): path.mkdir()

    # go through pose files
    # if distance is less than 120, copy all files with same name to output

    for pose_file in path_to_input_split.glob('poses/*.txt'):
        T = np.loadtxt(pose_file)

        position = T[:3, 3]
        distance = np.linalg.norm(position)

        print(distance)


        if distance < 140:
            i += 1
            name = pose_file.stem
            ext = pose_file.suffix

            # copy files from input to output
            # 1. poses
            pose_file_output = path_to_output_split / 'poses' / (name + '.txt')
            pose_file_output.write_bytes(pose_file.read_bytes())

            # 2. rgb
            rgb_file = path_to_input_split / 'rgb' / (name +  '.jpg')
            rgb_file_output = path_to_output_split / 'rgb' / (name + '.jpg')
            rgb_file_output.write_bytes(rgb_file.read_bytes())

            # 3. calibration
            calibration_file = path_to_input_split / 'calibration' / (name + '.txt')
            calibration_file_output = path_to_output_split / 'calibration' / (name + '.txt')
            calibration_file_output.write_bytes(calibration_file.read_bytes())

            # 4. init
            init_file = path_to_input_split / 'init' / (name + '.dat')
            init_file_output = path_to_output_split / 'init' / (name + '.dat')
            init_file_output.write_bytes(init_file.read_bytes())

print(i)