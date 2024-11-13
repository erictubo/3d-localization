# 3D Localization: Visual Localization against 3D models

This repository implements data generation from 3D models, particularly rendering CAD meshes in Blender and reading from COLMAP SFM models, as well as data conversion for localization pipelines MeshLoc and GLACE.

**Contents:**

...

## 1. Installation

### Blender (required for rendering)

- Install Blender from [blender.org](https://www.blender.org/download/) (tested with versions 4.1 and 4.2 LTS)

- Configure Blender task in [.vscode/tasks.json](.vscode/tasks.json): change command to location of Blender executable, e.g. "/snap/bin/blender" for Linux if installed via Snap or find the path by running `which blender` in terminal, to enable running Python rendering scripts through Blender directly from VS Code.

    ```json
    "command": "/Applications/Blender.app/Contents/MacOS/Blender",
    ```

To run a script with the Blender configuration, open the script in VS Code and press `Cmd/Ctrl + Shift + B`. This is only relevant for the files in the `blender/` directory, particularly [render_database.py](src/blender/render_database.py) for rendering automatic orbit poses, [render_query.py](src/blender/render_query.py) for rendering from COLMAP model poses, and [renderer.py](src/blender/renderer.py) for Blender rendering functions used by the prior two scripts.

The main script [main.py](src/main.py), which needs to use a different Python environment, integrates rendering COLMAP poses via the [interface_blender.py](src/interface_blender.py) script that interfaces with [render_query.py](src/blender/render_query.py) by running a terminal command.

### Python dependencies

HLoc has been used as the Conda environment for this repository as well.

1. set up hloc with its Python environment
2. Install additional dependencies

- pyquaternion
- transformations
- openexr (for depth maps)
- torch (for GLACE conversion)


HLoc (required by MeshLoc for feature extraction)

Hloc as submodule for global feature extraction required by MeshLoc

1. Follow hloc setup steps
2. Use hloc conda environment + install additional dependencies

```​
conda activate hloc_env

pip install pyquaternion transformations openexr
```

### MeshLoc & Immatch

MeshLoc: URL

Image Matching Toolbox: URL

- Python 3.8
- pycolmap
- cython
- ...




### 2. Datasets

[CadLoc](https://github.com/v-pnk/cadloc): [datasets](https://v-pnk.github.io/cadloc/datasets.html)


- Reference reconstruction
- CAD models

Download [IMC Phototourism](https://www.cs.ubc.ca/research/image-matching-challenge/2021/data/) datasets of reconstructed images from: https://www.cs.ubc.ca/%7Ekmyi/imw2020/data.html

- Notre Dame
- Pantheon
- Brandenburg Gate
- Reichstag

| Dataset | Images | 3D Points | CAD Model 1 | CAD Model 2 |
| --- | --- | --- | --- | --- |
| [Notre Dame](https://www.cs.ubc.ca/research/kmyi_data/imw2020/TrainingData/notre_dame_front_facade.tar.gz) | 3765 | 488 895 | [Notre Dame B](https://sketchfab.com/3d-models/notre-dame-de-paris-cbe2bbde869c4139912ce2cc35567d2c) | [Notre Dame E](https://www.myminifactory.com/object/3d-print-91899) |
| [Pantheon](https://www.cs.ubc.ca/research/kmyi_data/imw2020/TrainingData/pantheon_exterior.tar.gz) | 1401 | 166,923| [Pantheon B](https://3dwarehouse.sketchup.com/model/6bc6136fcb10d7ede8dcb1e1434763b8/Pantheon-Santa-Maria-della-Rotonda-Roma-It) | [Pantheon C](https://3dwarehouse.sketchup.com/model/ueff7b4b8-ab2c-4390-9f7d-b22fe05b3563/Roman-Pantheon)
| [Reichstag](https://www.cs.ubc.ca/research/kmyi_data/imw2020/TrainingData/reichstag.tar.gz) | 75 | 17,823 | [Reichstag A](https://3dwarehouse.sketchup.com/model/85521eefb0fcdf26b6ecbbb2e4d05900/Reichstag-Berlin-DEM-DEUTSCHEN-VOLK) | [Reichstag B](https://3dwarehouse.sketchup.com/model/7bc921e29beb8b9bb6ecbbb2e4d05900/Reichstag) |
| [Brandenburg Gate](https://www.cs.ubc.ca/research/kmyi_data/imw2020/TrainingData/brandenburg_gate.tar.gz) | 1363 | 100,040 | [Brandenburg Gate B](https://3dwarehouse.sketchup.com/model/e0dc0bb32222c0bb6ecbbb2e4d05900/Brandenburger-Tor) | [Brandenburg Gate C](https://3dwarehouse.sketchup.com/model/40b51e362efce960370a2be678914a9e/Brandenburg-Gate) |


Links to CAD models used

Process: export from SketchUp as OBJ, then import into Blender
Rename to "Model" and save as .blend file

Blend files added to repository

Before rendering, open the .blend file in Blender to check.

If textures are missing (model looks pink in textured view), go to File > External Data > Find Missing Files and select the textures folder.


Download ...
Registration ...
Blender import & settings ...

- Name "Model"

```text
- 3D Models
    - [Model Name]
        - Reference
            - dense
                - sparse
                    - cameras.txt / cameras.bin
                    - images.txt / images.bin
                    - points3D.txt / points3D.bin
                - images
        - [CAD Model ID]
            - [CAD Model ID].blend

- Evaluation
    - [Model Name] [CAD Model ID]
        - ground truth
            - T_sfm_cad.txt
        - inputs
            - database
            - query
        - outputs
```



## 3. Usage

### Main: Rendering Ground Truth Poses from Reconstruction

Set data paths in [data.py](src/data.py)

[main.py](src/main.py)

1. Ground truth conversion and rendering
2. Image Retrieval for MeshLoc (requires HLoc)
3. Evaluation of MeshLoc results




### Rendering Orbit Poses
[render_database.py](src/blender/render_database.py) – Rendering using automatic orbit poses

[blender/data.py](src/blender/data.py) to set data paths



### Visualization: Overlays, Depth Maps, Scene Coordinates
[visualization.py](src/visualization.py) – Visualization functions for overlays, depth maps, and scene coordinates


### GLACE Conversion
[glace_conversion.py](src/glace_conversion.py) – Data conversion for GLACE, reading from rendered data and COLMAP model files





## 4. Files Overview

### Main files

| File | Description |
| --- | --- |
| [main.py](src/main.py) | Main file for rendering and data conversion |
| [colmap_model.py](src/colmap_model.py) | Reading specific data from COLMAP model files and writing to text files, vice versa |
| [data.py](src/data.py) | Data structure: paths and file names for 3D models |
[features.py](src/features.py) | Feature extraction and matching for MeshLoc, using HLoc submodule |
| [glace_conversion.py](src/glace_conversion.py) | Data conversion for GLACE, reading from rendered data and COLMAP model files |
| [model_conversion.py](src/model_conversion.py) | Conversion between coordinate frames of CAD and COLMAP models, between data formats (e.g. depth maps) |
| [visualization.py](src/visualization.py) | Visualization functions for overlays, depth maps, and scene coordinates |

### Interface files

| File | Description |
| --- | --- |
| [interface_blender.py](src/interface_blender.py) | Blender interface for rendering, used by [main.py](src/main.py) for rendering of COLMAP poses |

### Blender files

Run these files using the Blender configuration in [.vscode/tasks.json](.vscode/tasks.json). In VS Code, once the task is configured, this can be done by pressing `Cmd/Ctrl + Shift + B`.

| File | Description |
| --- | --- |
| [blender/render_database.py](src/blender/render_database.py) | Rendering using automatic orbit poses |
| [blender/render_query.py](src/blender/render_query.py) | Rendering from COLMAP model poses |
| [blender/renderer.py](src/blender/renderer.py) | Blender rendering functions |
| [blender/data.py](src/blender/data.py) | Data paths |
| [.vscode/tasks.json](.vscode/tasks.json) | Task definition for Blender rendering |

### COLMAP files

| File | Description |
| --- | --- |
| [colmap/read_write_model.py](src/colmap/read_write_model.py) | Reading and writing COLMAP model files (copied from COLMAP repository) |







## 5. Data Structures

## A. Data Preparation

### Rendering

Format: Blender CAD

#### Input

3D Models

- Image database with COLMAP SFM model in `inputs/database/images`
- Blender file with CAD model in
- Registration matrix between SFM reference and CAD model placed in `ground truth / T_sfm_cad.txt`


#### Output: Data Structure

- images/*.png – rendered images
- depth/*.exr – depth maps (EXR format)
- intrinsics/*.txt – intrinsics (list format: `w, h, f, f_unit, cx, cy`)
- poses/*.txt – poses (Blender CAD frame)
- bounding_box – CAD model 3D corner points (CAD frame) for data verification

When rendering ground truth query poses from the main file ([main.py](src/main.py)), the output is saved to `ground truth/renders/`.

### Data Conversion for Localization

Localization in COLMAP SFM format

Conversion via transformation matrix $T_{ref}$ 

#### Manual Changes

1. rename depth files by replacing trailing "0001.exr" with ".exr" (name automatically generated by Blender)
2. Move query images to `inputs/query/images/*`
3. Move registration matrix to `ground truth / T_sfm_cad.txt`

#### Data Conversion Output

- input/
    - database/
        - depth/*.npz – depth maps (NPZ format)
        - cameras.txt – intrinsics (COLMAP format)
        - images.txt – poses (COLMAP format, SFM frame)
        - points3D.txt – CAD model corner points (COLMAP format, SFM frame) for data verification
    - query/
        - queries.txt – list of queries with intrinsics (for MeshLoc)
- output/
    - features/
        - pairs-from-retrieval.txt – top K matches from image retrieval (via HLoc)
        - pairs-from-retrieval-meshloc.txt – " (for MeshLoc)

#### Data Conversion for GLACE

Use script glace_conversion.py ...


## B. Localization Pipelines

### MeshLoc (Mesh-based Localization)

#### Parameters

- K for number of top matches to use: e.g. 25
- Method: e.g. patch2pix
- Config: e.g. aachen_v1.1
- ...

#### Data Structure

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
    - meshloc_out/
        - poses.txt – list of queries with output poses (SFM frame) (quaternion, translation)

#### MeshLoc Output: Data Structure

### GLACE (Scene Coordinate Regression)

#### Parameters

#### Data Structure

Pre-trained image retrieval model

Reading data based on sorted sequence, not by looking up filenames

- test/
    - calibration/*.txt – intrinsics (camera matrix format)
    - poses/*.txt – poses (transformation matrix format)
    - rgb/*.png – images
    - features.npy – global features extracted via R2Former
- train/
    - "
- output
    - <scene>.pt


## C. Post-Processing

### Data Conversion for Evaluation

- output/
    - cad_cam_poses.txt – list of queries with camera poses (CAD frame) (translation, quaternion)

### Rendering & Overlays

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

### Evaluation