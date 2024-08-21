# 3D Localization

## A. Data Preparation

### Rendering

Format: Blender CAD

#### Input

3D Models

- Image database with COLMAP SFM model
- Blender file with CAD model
- T_cad_sfm registration matrix between CAD and SFM model

For database rendering: set orbit parameters (e.g. distances, heights, angles, focal lengths)

For query rendering: read poses and intrinsics from COLMAP SFM model

#### Rendering Output: Data Structure

- images/*.png – rendered images
- depth/*.exr – depth maps (EXR format)
- intrinsics/*.txt – intrinsics (list format: `w, h, f, f_unit, cx, cy`)
- poses/*.txt – poses (Blender CAD frame)
- bounding_box – CAD model 3D corner points (CAD frame) for data verification

### Data Conversion for Localization

Localization in COLMAP SFM format

Conversion via transformation matrix $T_{ref}$ 

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