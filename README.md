# 3D Localization: Synthetic Data Generation from CAD Models for Visual Localization

[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Blender](https://img.shields.io/badge/Blender-4.1%2B-orange.svg)](https://www.blender.org/)

Generate synthetic training data for visual localization by rendering CAD models in Blender. This repository provides tools for:

1. Rendering images, depth maps, and scene coordinates from CAD models
    - Ground truth poses from SfM reconstruction
    - Automatic orbit poses around the model
2. Visualizing generated data (depth maps, scene coordinates) and comparing with ground truth data (overlays)
3. Reading and converting between different data formats (COLMAP, [MeshLoc](https://github.com/tsattler/meshloc_release), [GLACE](https://github.com/cvg/glace)) for localization pipelines

This repository is part of my [Master Thesis](https://github.com/erictubo/Master-Thesis) about visual localization against 3D models (see project page for report and presentation). The second component is [GLACE-3D](https://github.com/erictubo/glace-3d), a separate repository that adapts scene coordinate regression to 3D models (via supervised training against known scene coordinates and transfer learning for domain adaptation across real and synthetic data). Synthetic data generated from this repository is used for training GLACE-3D.

**Automatic Orbit Renders:** Images, depth maps, and scene coordinates

![automatic rendering](/preview/automatic_rendering.png)

**Overlays:** rendered SfM ground truth poses vs. real images

![overlays](/preview/overlays.png)
(accuracy limited by CAD model quality, SfM reconstruction, and CAD-SfM registration)

**Contents:**

1. [Installation](#1-installation)
    - [Blender](#blender)
    - [Python Environment (without MeshLoc)](#python-environment-without-meshloc)
    - [Python Environment (for MeshLoc)](#python-environment-for-meshloc)

2. [Datasets](#2-datasets)
    - [Original Datasets](#original-datasets)
    - [Quick Setup](#quick-setup)
    - [CAD Model Setup in Blender](#cad-model-setup-in-blender)
    - [Registration between SfM and CAD Model](#registration-between-sfm-and-cad-model)

3. [Usage](#3-usage)
    - [Rendering Ground Truth Poses](#rendering-ground-truth-poses)
    - [Rendering Automatic Poses](#rendering-automatic-poses)
    - [Visualization: Overlays, Depth Maps, Scene Coordinates](#visualization-overlays-depth-maps-scene-coordinates)
    - [Localization with MeshLoc (via ImMatch)](#localization-with-meshloc-via-immatch)
    - [Conversion for Localization with GLACE](#conversion-for-localization-with-glace)

4. [Files Overview](#4-files-overview)

5. [Testing](#testing)

6. [Acknowledgments](#acknowledgments)

## 1. Installation

Install Blender and Python environment with dependencies.

### Blender

Required for rendering

- Install Blender from [blender.org](https://www.blender.org/download/) (tested with versions 4.1 and 4.2 LTS)

- Configure Blender task in [.vscode/tasks.json](.vscode/tasks.json) to enable Python rendering scripts through Blender directly from VS Code: change command to location of Blender executable, e.g. "/snap/bin/blender" for Linux if installed via Snap or find the path by running `which blender` in terminal.

To run a script with the Blender configuration in VS Code, open the script and press `Cmd/Ctrl + Shift + B`. This is only relevant for the files in the `blender/` directory, particularly [render_database.py](src/blender/render_database.py) for rendering automatic poses, [render_query.py](src/blender/render_query.py) for rendering specific poses, and [renderer.py](src/blender/renderer.py) for rendering functions used by the prior two scripts.

The main script [main.py](src/main.py), which needs to use a different Python environment, integrates rendering the specified COLMAP poses via the [interface_blender.py](src/interface_blender.py) script that interfaces with [render_query.py](src/blender/render_query.py) by running a terminal command.

### Python Environment (without MeshLoc)

Create a Python environment with the required dependencies
(Python 3.8 has been used for compatibility with HLoc)

```bash
conda create -n 3d_env python=3.8

pip install numpy matplotlib opencv-python torch scipy pyquaternion transformations openexr
```

### Python Environment (for MeshLoc)

If [MeshLoc](https://github.com/tsattler/meshloc_release) is used for localization, the setup requires [HLoc](https://github.com/cvg/Hierarchical-Localization/) for feature extraction and matching as preparation for MeshLoc.
It can be combined in the same environment:

1. Create environment using Python 3.8

```bash
conda create -n hloc_env python=3.8
conda activate hloc_env
```

2. Clone HLoc and its dependencies

```bash
git clone --recursive https://github.com/cvg/Hierarchical-Localization/
cd Hierarchical-Localization/
python -m pip install -e .
```

3. Install additional dependencies

```bash
pip install pyquaternion transformations openexr
```

Alternatively, HLoc can be added as a submodule to this repository. If feature extraction for MeshLoc is not used, HLoc is not required.

## 2. Datasets

For each dataset, we need:

- Real images & SfM reconstruction as reference
- Corresponding CAD model(s)
- Registration matrix between SfM reconstruction and CAD model

### Original Datasets: Images & SfM Reconstruction

Download [IMC Phototourism](https://www.cs.ubc.ca/research/image-matching-challenge/2021/data/) datasets of reconstructed images from: https://www.cs.ubc.ca/%7Ekmyi/imw2020/data.html

Most datasets and registration matrices have been used from [CadLoc](https://github.com/v-pnk/cadloc) / [Datasets](https://v-pnk.github.io/cadloc/datasets.html), which also made use of IMC Phototourism datasets.

| Dataset (Images + SfM Reconstruction) | # Images | # 3D Points | CAD Model 1 | CAD Model 2 |
| --- | --- | --- | --- | --- |
| [Notre Dame](https://www.cs.ubc.ca/research/kmyi_data/imw2020/TrainingData/notre_dame_front_facade.tar.gz) | 3765 | 488 895 | [Notre Dame B](https://sketchfab.com/3d-models/notre-dame-de-paris-cbe2bbde869c4139912ce2cc35567d2c) | [Notre Dame E](https://www.myminifactory.com/object/3d-print-91899) |
| [Pantheon](https://www.cs.ubc.ca/research/kmyi_data/imw2020/TrainingData/pantheon_exterior.tar.gz) | 1401 | 166,923| [Pantheon B](https://3dwarehouse.sketchup.com/model/6bc6136fcb10d7ede8dcb1e1434763b8/Pantheon-Santa-Maria-della-Rotonda-Roma-It) | [Pantheon C](https://3dwarehouse.sketchup.com/model/ueff7b4b8-ab2c-4390-9f7d-b22fe05b3563/Roman-Pantheon)
| [Reichstag](https://www.cs.ubc.ca/research/kmyi_data/imw2020/TrainingData/reichstag.tar.gz) | 75 | 17,823 | [Reichstag A](https://3dwarehouse.sketchup.com/model/85521eefb0fcdf26b6ecbbb2e4d05900/Reichstag-Berlin-DEM-DEUTSCHEN-VOLK) | [Reichstag B](https://3dwarehouse.sketchup.com/model/7bc921e29beb8b9bb6ecbbb2e4d05900/Reichstag) |
| [Brandenburg Gate](https://www.cs.ubc.ca/research/kmyi_data/imw2020/TrainingData/brandenburg_gate.tar.gz) | 1363 | 100,040 | [Brandenburg Gate B](https://3dwarehouse.sketchup.com/model/e0dc0bb32222c0bb6ecbbb2e4d05900/Brandenburger-Tor) | [Brandenburg Gate C](https://3dwarehouse.sketchup.com/model/40b51e362efce960370a2be678914a9e/Brandenburg-Gate) |

### Quick Setup

Readily prepared data for the CAD models of the above datasets are available in the [models directory](models), including:

- BLEND files for Blender rendering
- Textures for shading in Blender
- Registration matrices: `T_ref.txt` (CAD model in the frame of the SfM model)

Images & reconstructions still need to be downloaded from the links in the first column. These are too large to include in the repository and don't need to be reformatted anyway.

The next steps can be skipped if using the provided data.

### CAD Model Setup in Blender

After downloading the CAD model:

- Make sure the units of the model are correct (e.g. meters) – e.g. in SketchUp this can be changed before exporting: Model Info > Units > Decimal – note that SketchUp requires a Pro license (or free trial) for exporting.
- Import the CAD model into Blender - OBJ format (exported from SketchUp) often works better than Collada DAE
- Set name the name of the CAD import to "Model"
- Make sure there is a camera called "Camera" and a light called "Light" in the scene.
- Save the Blender file as `[Model Name] [CAD Model ID].blend` in the corresponding directory

Before rendering, open the BLEND file in Blender to check it looks. If textures are missing (model looks pink in textured view), go to File > External Data > Find Missing Files and select the textures folder. Ideally, place the textures in the same directory as the BLEND file in a folder called "textures". Then save the file again to remember the texture directory for rendering.

### Registration between SfM and CAD Model

Reference transformation matrix `T_ref.txt` or `T_sfm_cad.txt` (CAD model in the frame of the SfM model).

#### Create a new registration matrix

Using CloudCompare:

- Load the CAD model and the SfM reference model (e.g. import both as PLY, exported from Blender / COLMAP)
- Align the CAD model with the SfM reference – make sure that the CAD model is selected as to-be-aligned:
  - Edit > Translate/Rotate + Edit > Multiply/Scale
  - Tools > Registration > Align (point pairs picking)
  - Tools > Registration > Fine Registration (ICP)
- Save the transformation matrix as a text file (T_sfm_cad.txt) in the ground truth directory

#### Verify an existing registration matrix

Under CloudCompare > Edit > Transformation > Apply Transformation, select the CAD model and apply the transformation matrix to align the CAD model with the SfM reference.

## 3. Usage

Rendering options include ground truth poses from the SfM reconstruction and automatic poses around the model. Outputs can be visualized and converted for localization with MeshLoc or GLACE.

- [Rendering Ground Truth Poses](#rendering-ground-truth-poses)
- [Rendering Automatic Poses](#rendering-automatic-poses)
- [Visualization: Overlays, Depth Maps, Scene Coordinates](#visualization-overlays-depth-maps-scene-coordinates)
- [Localization with MeshLoc (via ImMatch)](#localization-with-meshloc-via-immatch)
- [Conversion for Localization with GLACE](#conversion-for-localization-with-glace)

### Rendering Ground Truth Poses

Steps of the main script ([main.py](src/main.py)) for rendering ground truth poses from an SfM reconstruction:

1. Ground truth conversion and rendering
2. Image Retrieval for MeshLoc – requires HLoc (optional)
3. Evaluation of MeshLoc results (optional)

#### Input

To run [main.py](src/main.py), set `path_to_data` in [data.py](src/data.py) and structure the directory as follows:

```text
- models
    - [Model Name]
        - Reference
            - dense
                - sparse
                    - cameras.txt / .bin
                    - images.txt / .bin
                    - points3D.txt / .bin
                - images
        - [CAD Model ID]
            - [Model Name] [CAD Model ID].blend
            - textures (for shading in Blender)

- evaluation
    - [Model Name] [CAD Model ID]
        - ground truth
            - T_sfm_cad.txt
        - inputs
            - database
            - query
        - outputs
```

Where `[Model Name]` is the name of the 3D model, e.g. "Notre Dame", and `[CAD Model ID]` is the ID of the CAD model, e.g. "B" or "E". The `cameras`, `images`, and `points3D` files are from the COLMAP SfM reconstruction, and `T_sfm_cad.txt` is the registration matrix (the CAD model in the frame of the SfM model).

#### Output

Step 1.1:
Under `inputs/`, `queries.txt` is saved with the list of all images from the COLMAP model and their intrinsics (format: `image_name camera_model w h fx fy cx cy`) used for MeshLoc.

Step 1.2-1.3:
In the `ground truth/` directory, the following files are also saved: `cam_sfm_poses.txt` (poses from COLMAP model, SfM in camera frame), `cad_cam_poses.txt` (converted camera poses in CAD frame).

Step 1.4:
Rendering ground truth query poses from the main file ([main.py](src/main.py)), the output is saved to `ground truth/renders/`. The output format is the following:

```text
- images/*.png – rendered images
- depth/*.exr – depth maps (EXR format) [m]
- intrinsics/*.txt – camera intrinsics: w, h, f, f_unit, cx, cy
- poses/*.txt – poses (Blender CAD frame): tx, ty, tz, qw, qx, qy, qz
- bounding_box – CAD model 3D corner points (CAD frame) for data verification
```

#### Depth Conversion & Scene Coordinates

The main file (step 1.5) will do this automatically for the rendered ground truth poses. Otherwise functions in [model_conversion.py](src/model_conversion.py) can be used to convert the EXR depth maps to NPZ format for MeshLoc (see implementation for rendering ground truth in [main.py](src/main.py)).

Often Blender will add a suffix to the depth map file name, e.g. "0001.exr", which needs to be removed manually such that names are detected correctly across the data.
To remove this suffix, navigate to the depth directory and use the following command:

```bash
cd <render_dir>/depth
for f in *; do mv -- "$f" "${f%0001.exr}.exr"; done
```

#### Image Retrieval for MeshLoc (Optional)

Step 2: if running the MeshLoc options in the main script, the following output is also saved:

```text
- output/
    - features/
        - pairs-from-retrieval.txt – top K matches from image retrieval (via HLoc)
        - pairs-from-retrieval-meshloc.txt – " (for MeshLoc)
```

Note that this requires HLoc (see installation).

#### Evaluation of Results from MeshLoc (Optional)

See step 4 of [main.py](src/main.py) for evaluation of MeshLoc results. It will generate the following additional outputs:

Step 4.1-4.2: Conversion

```text
- output/
    - cad_cam_poses.txt – list of queries with camera poses (CAD frame) (translation, quaternion)
```

Step 4.3: Renders & overlays

```text
- ground truth/
    - renders/
        - [see Rendering Output] for each query GT pose
    - overlays/
        - *.jpg – overlaid query and output image
- output
    - renders/
        - [see Rendering Output] for each query output pose
    - overlays/
        - *.jpg – overlaid query and rendered output pose
```

### Rendering Automatic Poses

- [render_database.py](src/blender/render_database.py) – Rendering using automatic orbit poses

- [blender/data.py](src/blender/data.py) to set data paths

#### Input

No SfM reconstruction required, only the CAD model:

```text
- models
    - [Model Name]
        - [CAD Model ID]
            - [Model Name] [CAD Model ID].blend
            - textures (for shading in Blender)
```

#### Settings

See `render_orbit_views()` or `render_ground_views()` in [renderer.py](src/blender/renderer.py)

`render_orbit_views()` generates orbit poses around the model with vertical angles specified.

```python
h_steps: int,                           # horizontal steps
v_angles_deg: 'list[int]',              # vertical angles in degrees (horizontal is zero)
distances: 'list[int]' = None,          # distances center of model
focal_lengths: 'list[float]' = [35],    # focal lengths
f_unit: str = 'MM'                      # focal unit, can be 'MM', 'PIX', or FoV: 'DEG', 'RAD'
```

`render_ground_views()` generates views around the model at specified heights above the ground, with `v_angle` calculated such that it automatically centers the model.

- Can lead to more realistic poses that are closer to the ground rather than aerial views.
- The offsets can be used to shift the camera position in the horizontal and vertical directions to introduce more variety.

```python
distances: 'list[int]',
h_steps: int,
heights: 'list[int]',                   # height above ground [m] instead of vertical angles
focal_lengths: 'list[float]' = [35],
f_unit: str = 'MM',
v_offsets_deg: 'list[int]' = [0],       # vertical offsets in degrees
h_offsets_deg: 'list[int]' = [0],       # horizontal offsets in degrees
```

#### Output

The outputs are saved to the `render_dir` specified in [blender/data.py](src/blender/data.py). By default, this is set to `renders/[Model Name] [CAD Model ID]/`, but could also be set to `evaluation/[Model Name]/[CAD Model ID]/input/database/` for direct use in MeshLoc.
The format is the same as for the ground truth rendering, with the rendered images, depth maps (EXR), intrinsics, and poses saved in separate directories.

### Visualization: Overlays, Depth Maps, Scene Coordinates

See [visualization.py](src/visualization.py) for functions to:

- create overlays of real and rendered images
- visualize depth maps
- visualize scene coordinates
- compare depth maps – pixelwise distance between two depth maps
- compare scene coordinates – pixelwise distance between two scene coordinate maps

Examples that have been tested are included at the end of the file in the `if __name__ == "__main__":` block.
In `visualize_depth_map()` and `visualize_scene_coordinate_map()`, some settings can be used to make the color maps uniform across different depth maps or scene coordinates (e.g. `depth_range` or `x_range`, `y_range`, `z_range`).

### Localization with MeshLoc (via ImMatch)

The output format of the main script is compatible with MeshLoc, but the installation needs to be followed such that the features are extracted and matched using HLoc.

#### Setup

To run MeshLoc:

1. Install Image Matching Toolbox: [Immatch](https://github.com/GrumpyZhou/image-matching-toolbox)

   - Python 3.8
   - pycolmap
   - cython
   - ...

2. Install MeshLoc: [MeshLoc](https://github.com/tsattler/meshloc_release)

#### MeshLoc Input

(automatically generated by the main script with MeshLoc setup)

The following data is required by MeshLoc:

```text
- input/
    - database/
        - images/*.png
        - depth/*.npz – depth maps (NPZ format)
        - cameras.txt – intrinsics (COLMAP format)
        - images.txt – poses (COLMAP format)
    - query/
        - images/
        - queries.txt
- output/
    - features/
        - retrieval-pairs.txt – global features from image retrieval step (via HLoc)
    - meshloc_matches/
        - *.npy – local feature matches for each query-database pair
```

Place any images to-be-localized in the `query/images` directory. The intrinsics (format: `image_name camera_model w h fx fy cx cy`) of all images from the reconstruction are saved in `queries.txt` (created by the main script). The name of the query image is used to look up the corresponding intrinsics.

#### MeshLoc Settings

- K for number of top matches to use: e.g. 25
- Method: e.g. patch2pix
- Config: e.g. aachen_v1.1
- ...

Example for running MeshLoc (from the ImMatch repository) with the above data structure:

```bash
python ../meshloc_release/localize.py \
--db_image_dir "<path_to_data>/Evaluation/notre dame B/inputs/database/images" \
--db_depth_image_dir "<path_to_data>/Evaluation/notre dame B/inputs/database/depth" \
--colmap_model_dir "<path_to_data>/Evaluation/notre dame B/inputs/database" \
--query_dir "<path_to_data>/Evaluation/notre dame B/inputs/query/images" \
--query_list "<path_to_data>/Evaluation/notre dame B/inputs/query/queries.txt" \
--match_prefix "<path_to_data>/Evaluation/notre dame B/outputs/meshloc_match/superglue/" \
--out_prefix "<path_to_data>/Evaluation/notre dame B/outputs/meshloc_out/superglue/" \
--method_name superglue \
--method_config aachen_v1.1 \
--method_string superglue_aachen_v1_1_ \
--retrieval_pairs "<path_to_data>/Evaluation/notre dame B/outputs/features/pairs-from-retrieval-meshloc.txt" \
--top_k 25 \
--max_side_length -1 \
--ransac_type POSELIB+REF \
--min_ransac_iterations 10000 \
--max_ransac_iterations 100000 \
--reproj_error 20.0 \
--use_orig_db_images \
--cluster_keypoints
```

#### MeshLoc Output

Will be saved to the `meshloc_out` directory:

```text
- output/
    - meshloc_out/
        - poses.txt – list of queries with output poses (SFM frame) (quaternion, translation)
```

### Conversion for Localization with GLACE

[glace_conversion.py](src/glace_conversion.py) – Data conversion for GLACE, reading from rendered data and COLMAP model files

Conversion of data for compatibility with GLACE Scene Coordinate Regression:

- Read from (A) CAD renders or (B) SFM reconstruction
- Write to `train/*`, `test/*`


```python
path_to_glace: Path,        # where to save the GLACE data
num_test: int = 0,          # number of test samples (uses first n samples)
target_height: int = 480,   # image height for resizing
```

#### Generate from Reconstruction (Real Data)

First, open the reconstruction in COLMAP (under File > Import Model), saved in `.../Reference/dense/sparse`) and export as an NVM file (under File > Export Model as).

Then, use the following settings to convert the data:

```python
path_to_nvm: Path,              # path to COLMAP NVM file
path_to_images: Path,           # path to images (.../Reference/dense/images)
T_sfm_cad: np.ndarray,          # registration matrix
depth_maps: bool = True,        # convert depth maps
scene_coordinates: bool = True, # convert scene coordinates
to_mm = True,                   # convert depth maps to mm (default GLACE format)
nn_subsampling = 8,             # subsampling for depth maps and scene coordinates (GLACE predicts at 1/8 resolution)
```

The registration matrix is used to convert the poses from the SfM frame to the CAD frame, which has realistic scaling.
For consistency and testing comparability, make sure tot use the registration matrix for the corresponding CAD model.
It is automatically reversed to `T_cad_sfm`.

#### Generate from CAD Renders

```python
path_to_renders: Path,          # path to rendered data with images, intrinsics, poses, depth maps
depth_maps: bool = True,        # "
to_mm = True,                   # "
nn_subsampling = 1,             # subsampling later done by GLACE, to maintain accuracy with augmentations
```

Scene coordinate maps are not created, because GLACE can create them from the non-subsampled depth maps more accurately, especially when using data augmentation (e.g. rotation) during training.

Blender coordinate frame with camera looking in -Z, so they are automatically reversed to +Z in the script (using `reverse_camera_pose_for_blender` from [model_conversion](src/model_conversion.py)). The output remains in the CAD frame.

#### Output

In both cases, the output is saved to the `path_to_glace` directory:

```text
- test/
    - calibration/*.txt – intrinsics (camera matrix format)
    - poses/*.txt – poses (transformation matrix format)
    - rgb/*.png – images
    - features.npy – global features extracted via R2Former
- train/
    - "
```

Reading data based on sorted sequence, not by looking up filenames

GLACE will use this data to train a scene-specific prediction head `<scene>.pt`

See the [GLACE](https://github.com/cvg/glace) or [GLACE-3D](https://github.com/erictubo/glace-3d) repository for more details.

## 4. Files Overview

### Main Files

| File | Description |
| --- | --- |
| [main.py](src/main.py) | Main file for rendering and data conversion |
| [colmap_model.py](src/colmap_model.py) | Reading specific data from COLMAP model files and writing to text files, vice versa |
| [data.py](src/data.py) | Data structure: paths and file names for 3D models |
| [features.py](src/features.py) | Feature extraction and matching for MeshLoc, using HLoc submodule |
| [glace_conversion.py](src/glace_conversion.py) | Data conversion for GLACE, reading from rendered data and COLMAP model files |
| [model_conversion.py](src/model_conversion.py) | Conversion between coordinate frames of CAD and COLMAP models, between data formats (e.g. depth maps) |
| [visualization.py](src/visualization.py) | Visualization functions for overlays, depth maps, and scene coordinates |

### Interface Files

| File | Description |
| --- | --- |
| [interface_blender.py](src/interface_blender.py) | Blender interface for rendering, used by [main.py](src/main.py) for rendering of COLMAP poses |

### Blender Files

Run these files using the Blender configuration in [.vscode/tasks.json](.vscode/tasks.json). In VS Code, once the task is configured, this can be done by pressing `Cmd/Ctrl + Shift + B`.

| File | Description |
| --- | --- |
| [blender/render_database.py](src/blender/render_database.py) | Rendering using automatic orbit poses |
| [blender/render_query.py](src/blender/render_query.py) | Rendering from COLMAP model poses |
| [blender/renderer.py](src/blender/renderer.py) | Blender rendering functions |
| [blender/data.py](src/blender/data.py) | Data paths |
| [.vscode/tasks.json](.vscode/tasks.json) | Task definition for Blender rendering |

### COLMAP Files

| File | Description |
| --- | --- |
| [colmap/read_write_model.py](src/colmap/read_write_model.py) | Reading and writing COLMAP model files (copied from COLMAP repository) |

## 5. Testing

Correctness in this project is primarily verified through visualization of outputs (e.g., overlays, depth maps, scene coordinates) and runtime assertions that check data shapes and file existence. Due to the nature of 3D rendering and data generation, traditional unit tests are not used, as outputs are best evaluated visually rather than by exact values.

## 6. Acknowledgments

- [MeshLoc](https://github.com/tsattler/meshloc_release) for the localization pipeline
- [GLACE](https://github.com/cvg/glace) for scene coordinate regression
- [COLMAP](https://github.com/colmap/colmap) for SfM reconstruction
