import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from matplotlib.patches import ConnectionPatch
import pycolmap
from pathlib import Path

from typing import Dict, Tuple, List, Optional, Union
from pprint import pformat
from types import SimpleNamespace

from hloc import extract_features, match_features, pairs_from_retrieval
from hloc.utils import read_write_model
from hloc.utils.io import read_image, get_keypoints, get_matches

# import os
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


class Features:

    def __init__(
            self,
            images: Path,
            outputs: Path,
            retrieval_pairs: Path,
            global_feature_conf_name: str,
            global_num_matched: int,
            local_feature_conf_name: str,
            local_match_conf_name: str,
            db_prefix: str = 'database',
            query_prefix: str = 'query'
        ) -> None:

        # Path to images and outputs directory
        self.images = images
        self.outputs = outputs

        # Path to retrieval pairs file
        self.retrieval_pairs = retrieval_pairs
        if not retrieval_pairs.exists(): retrieval_pairs.touch()
        
        # Global feature configuration
        assert global_feature_conf_name in extract_features.confs,\
            f"Invalid feature configuration name: {global_feature_conf_name}\
            \nSelect from:\n{pformat(extract_features.confs)}"
        self.global_feature_conf = extract_features.confs[global_feature_conf_name]

        # Local feature configuration
        assert local_feature_conf_name in extract_features.confs,\
            f"Invalid feature configuration name: {local_feature_conf_name}\
            \nSelect from:\n{pformat(extract_features.confs)}"
        self.local_feature_conf = extract_features.confs[local_feature_conf_name]

        # Local matching configuration
        assert local_match_conf_name in match_features.confs,\
            f"Invalid feature configuration name: {local_match_conf_name}\
            \nSelect from:\n{pformat(match_features.confs)}"
        self.local_match_conf = match_features.confs[local_match_conf_name]

        self.global_num_matched = global_num_matched

        self.db_prefix = db_prefix
        self.query_prefix = query_prefix

    
    def image_retrieval(self) -> Path:
        """
        Retrieve image pairs using global feature matching.
        """
        global_descriptors = extract_features.main(
            conf=self.global_feature_conf,
            image_dir=images,
            export_dir=outputs,
        )        

        pairs_from_retrieval.main(
            descriptors=global_descriptors,
            output=retrieval_pairs,
            num_matched=self.global_num_matched,
            db_prefix=self.db_prefix,
            query_prefix=self.query_prefix,
        )

        return global_descriptors


    def local_feature_matching(self) -> Tuple[Path, Path]:
        """
        Match local features between query and database images.
        """
        
        local_features = extract_features.main(
            conf=self.local_feature_conf,
            image_dir=self.images,
            export_dir=self.outputs,
        )    

        local_matches = match_features.main(
            conf=self.local_match_conf,
            pairs=self.retrieval_pairs,
            features=self.local_feature_conf['output'],
            export_dir=self.outputs,
        )

        return local_features, local_matches


    def check_retrieval_pairs(self, query_name: str, db_names: List[str] = None) -> bool:
        """
        Check that retrieval pairs exist for query image and database images.
        """
        with open(self.retrieval_pairs, 'r') as f:
            for db_name in db_names:
                if not any(line.startswith(query_name) and db_name in line for line in f):
                    raise ValueError(f"Pair not found: {query_name} {db_name}")

    def get_retrieved_db_images(self, query_name: str) -> List[str]:
        """
        Get retrieved database images for query image.
        """
        with open(self.retrieval_pairs, 'r') as f:
            db_names = [line.split()[1] for line in f if line.startswith(query_name)]
        return db_names

    def visualize_local_matches(self, query_name: str, db_names: List[str] = None, db_limit: int = 10) -> None:
        '''
        Visualize local matches between query image and paired database images.
        Draw points and lines between matched keypoints, with query image on the left, and database image on the right.
        '''

        if not query_name.startswith(self.query_prefix):
            query_name = self.query_prefix + '/' + query_name
        query_image = read_image(images / query_name)

        if db_names: self.check_retrieval_pairs(query_name, db_names)  
        else: db_names = self.get_retrieved_db_images(query_name)

        for i, db_name in enumerate(db_names):
            i += 1
            if i > db_limit: break

            if not db_name.startswith(self.db_prefix):
                db_name = self.db_prefix + '/' + db_name
            db_image = read_image(images / db_name)
            # pair, reverse = find_pair(local_matches, query_name, db_name)
            # if pair:
            matches, scores = get_matches(local_matches, query_name, db_name)

            query_keypoints = get_keypoints(local_features, query_name)
            db_keypoints = get_keypoints(local_features, db_name)

            # Visualize matches
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            ax[0].imshow(query_image)
            ax[1].imshow(db_image)
            for i, (q, db) in enumerate(matches):
                if scores[i] < 0.7: continue
                # set new color for each match
                c = plt.cm.jet(i / len(matches))
                # query image left
                ax[0].set_title(query_name)
                ax[0].scatter(query_keypoints[q, 0], query_keypoints[q, 1], c=c, s=5)
                # database image right
                ax[1].set_title(db_name)
                ax[1].scatter(db_keypoints[db, 0], db_keypoints[db, 1], c=c, s=5)
                # connect keypoints with lines using ConnectionPatch
                con = ConnectionPatch(xyA=(query_keypoints[q, 0], query_keypoints[q, 1]), xyB=(db_keypoints[db, 0], db_keypoints[db, 1]),
                                        coordsA="data", coordsB="data", axesA=ax[0], axesB=ax[1], color=c)
                ax[1].add_artist(con)

            plt.show()


class ColmapData:
    """
    Data retrieval class for COLMAP SfM ground truth database.
    """

    def __init__(self, sfm_model: Path):

        self.sfm_images_dict = read_write_model.read_images_binary(sfm_model / 'dense/sparse/' / 'images.bin')

        # images[image_id] = Image(
        #             id=image_id,
        #             qvec=qvec,
        #             tvec=tvec,
        #             camera_id=camera_id,
        #             name=image_name,
        #             xys=xys,
        #             point3D_ids=point3D_ids,
        #         )

        self.sfm_cameras_dict = read_write_model.read_cameras_binary(sfm_model / 'dense/sparse/' / 'cameras.bin')

        # cameras[camera_id] = Camera(
        #             id=camera_id,
        #             model=model_name,
        #             width=width,
        #             height=height,
        #             params=np.array(params),
        #         )

       # self.sfm_points_dict = read_write_model.read_points3d_binary(sfm_model / 'dense/sparse/' / 'points3D.bin')

        # points3D[point3D_id] = Point3D(
        #         id=point3D_id,
        #         xyz=xyz,
        #         rgb=rgb,
        #         error=error,
        #         image_ids=image_ids,
        #         point2D_idxs=point2D_idxs,
        #     )

    def get_image_id_from_name(self, image_name: str) -> int:
        """
        Get image ID from image name.
        """
        image_name = image_name.split('/')[-1]
        image_id = [k for k, v in self.sfm_images_dict.items() if v.name == image_name][0]
        return image_id
    
    def get_pose_from_id(self, image_id: int) -> np.ndarray:
        """
        Get pose of image from database.
        """
        sfm_image = self.sfm_images_dict[image_id]
        pose = np.concatenate([sfm_image.tvec, sfm_image.qvec])
        assert pose.shape == (7,), pose.shape
        return pose

    def get_pose_from_name(self, image_name: str) -> np.ndarray:
        """
        Get pose of image from database.
        """
        image_id = self.get_query_sfm_id(image_name, self.sfm_images_dict)
        return self.get_pose_from_id(image_id)
    
    def get_intrinsics_from_id(self, image_id: int) -> Tuple[str, np.ndarray, int, int]:
        """
        Get intrinsics of image from database.
        Output: camera model, camera parameters, image width, image height
        """
        sfm_image = self.sfm_images_dict[image_id]
        camera_id = sfm_image.camera_id

        sfm_camera = self.sfm_cameras_dict[camera_id]

        camera_model: str = sfm_camera.model
        camera_params: np.ndarray = sfm_camera.params
        camera_width: int = sfm_camera.width
        camera_height: int = sfm_camera.height

        return camera_model, camera_params, camera_width, camera_height

    def get_intrinsics_from_name(self, image_name: str) -> Tuple[str, np.ndarray, int, int]:
        """
        Get intrinsics of query image from COLMAP SfM ground truth database.
        Output: camera model, camera parameters, image width, image height
        """
        image_id = self.get_query_sfm_id(image_name, self.sfm_images_dict)
        return self.get_intrinsics_from_id(image_id)


# POSE RETRIEVAL & TRANSFORMATION

def get_db_pose(db_name):
    """
    Get pose of database image from render output text file.
    Format: scalar-first (px, py, pz, qw, qx, qy, qz)
    """
    pose_name = db_name.replace('.png', '.txt').replace('image', 'pose')
    pose = np.loadtxt(images / pose_name)

    return pose


def pose_to_matrix(pose: np.ndarray) -> np.ndarray:
    """
    Convert pose vector to transformation matrix.
    Format: scalar-first (p_x, p_y, p_z, q_w, q_x, q_y, q_z)
    """
    t = pose[:3]
    q = pose[3:]
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix() # scalar-last in scipy
    T[:3, 3] = t

    return T

def matrix_to_pose(T: np.ndarray) -> np.ndarray:
    """
    Convert transformation matrix to pose vector (p_x, p_y, p_z, q_x, q_y, q_z, q_w).
    """
    t = T[:3, 3]
    q = Rotation.from_matrix(T[:3, :3]).as_quat() # scalar-last
    q = np.concatenate([q[1:], q[:1]]) # scalar-first
    pose = np.concatenate([t, q])

    return pose

def invert_matrix(T: np.ndarray) -> np.ndarray:
    """
    Invert transformation matrix.
    """
    assert T.shape == (4,4), T.shape
    R = T[:3, :3]
    t = T[:3, 3]
    R_inv = R.transpose()
    t_inv = -R_inv @ t
    T_inv = np.eye(4)
    T_inv[:3, :3] = R_inv
    T_inv[:3, 3] = t_inv

    return T_inv

def get_reference_transformation():
    """
    Get reference transformation from text file.
    """
    name = 'reference_transformation.txt'
    T_ref = np.loadtxt(images / name)
    return T_ref


def save_to_file(name: str, data: np.ndarray, outputs: Path):
    """
    Save data to file.
    """
    with open(outputs / name, 'w') as f:
            f.write(' '.join(map(str, data)))




# RENDERING
# TODO: render white images with shading only
# TODO: render CAD model from image pose ... good for testing, feature matching & evaluation later on


# EVALUATION
# - global retrieval accuracy
# - local feature matching accuracy (needs localization OR ground truth to determine outliers)
# - localization accuracy


# LOCALIZATION
# - TODO: integrate MeshLoc with my depth data


if __name__ == '__main__':

    dataset = Path('/Users/eric/Downloads/evaluation/model_B/')
    images = dataset / 'images/'
    outputs = dataset / 'outputs/'

    retrieval_pairs = outputs / "pairs-from-retrieval.txt"

    features = Features(
        images=images,
        outputs=outputs,
        retrieval_pairs=retrieval_pairs,
        global_feature_conf_name='netvlad',
        global_num_matched=10,
        local_feature_conf_name='superpoint_aachen',
        local_match_conf_name='superglue',
        )

    global_descriptors = features.image_retrieval()
    local_features, local_matches = features.local_feature_matching()


    # COLMAP SFM Model
    sfm_model = Path('/Users/eric/Downloads/notre_dame_front_facade')
    colmap_data = ColmapData(sfm_model)





    # from registration in Blender
    # pose_sfm_cad = np.array([-120.84, 55.401, 8.9346, 0.388, -0.274, 0.528, -0.697])
    # scale_sfm_cad = 11


    # T_cad_sfm = get_reference_transformation()
    # T_sfm_cad = invert_matrix(T_cad_sfm)
    

    #print('Reference transformation: ', T_cad_sfm)

    # pose_cad_sfm = matrix_to_pose(T_cad_sfm)

    # print('Reference pose (SfM): ', pose_cad_sfm)


    # From CloudCompare registration
    # T_sfm_cad = np.array([
    #     [-5.636510, 3.172812, 9.269413, -123.011314],
    #     [-9.796731, -1.701635, -5.374708, 51.065014],
    #     [-0.113225, -10.714321, 3.598537, 6.478726],
    #     [0.000000, 0.000000, 0.000000, 1.000000]
    # ])

    # print('Transformation matrix (SfM in CAD): ', T_sfm_cad)

    pose_sfm_cad = np.array([-123.6, 52.301, 2.6172, -0.408, 0.280, -0.492, 0.715])
    scale_sfm_cad = 11.500


    T_sfm_cad = pose_to_matrix(pose_sfm_cad)


    print('Pose (SfM in CAD): ', pose_sfm_cad)


    query_names=[
        '01333333_3920521666.jpg',
        # '00870470_3859452456.jpg',
        # '05353713_5401929533.jpg',
        ]

    for query_name in query_names:
        query_id = colmap_data.get_image_id_from_name(query_name)

        query_intrinsics = colmap_data.get_intrinsics_from_id(query_id)
        pose_query_sfm = colmap_data.get_pose_from_id(query_id)

        T_query_sfm = pose_to_matrix(pose_query_sfm)
        T_query_cad = T_query_sfm @ T_sfm_cad
        pose_query_cad = matrix_to_pose(T_query_cad)

        print('Query intrinsics: ', query_intrinsics)
        print('Query pose (SfM): ', pose_query_sfm)
        print('Query transformation matrix (SfM): ', T_query_sfm)
        print('Query transformation matrix (CAD): ', T_query_cad)
        print('Query pose (CAD): ', pose_query_cad)

        save_to_file(query_name.replace('.jpg', '.txt').replace('query/',''), pose_query_cad, outputs)

        # pose_db_cad = get_db_pose('database/image_0040.png')
        # T_db_cad = pose_to_matrix(pose_db)

        features.visualize_local_matches(query_name, db_limit=1)

        


# NEXT:
# - render CAD model from query pose
# - combine rendered image with query image

# - run Blender code without render task configuration


# - make MeshLoc compatbile with my depth data (npy/npz files)