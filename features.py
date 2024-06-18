from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from typing import Dict, Tuple, List, Optional, Union
from pprint import pformat
from types import SimpleNamespace

from hloc import extract_features, match_features, pairs_from_retrieval
from hloc.utils.io import read_image, get_keypoints, get_matches

# import os
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


class Features:

    def __init__(
            self,
            path_to_images: Path,
            path_to_outputs: Path,
            path_to_retrieval_pairs: Path,
            global_feature_conf_name: str,
            global_num_matched: int,
            local_feature_conf_name: str,
            local_match_conf_name: str,
            db_prefix: str = 'database',
            query_prefix: str = 'query'
        ):

        # Path to images and outputs directory
        self.path_to_images = path_to_images
        self.path_to_outputs = path_to_outputs

        # Path to retrieval pairs file
        self.path_to_retrieval_pairs = path_to_retrieval_pairs
        if not path_to_retrieval_pairs.exists(): path_to_retrieval_pairs.touch()
        
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

        self.path_to_global_descriptors = None
        self.path_to_local_features = None
        self.path_to_local_matches = None

    
    def image_retrieval(self):
        """
        Retrieve image pairs using global feature matching.
        """
        self.path_to_global_descriptors = extract_features.main(
            conf=self.global_feature_conf,
            image_dir=self.path_to_images,
            export_dir=self.path_to_outputs,
        )        

        pairs_from_retrieval.main(
            descriptors=self.path_to_global_descriptors,
            output=self.path_to_retrieval_pairs,
            num_matched=self.global_num_matched,
            db_prefix=self.db_prefix,
            query_prefix=self.query_prefix,
        )


    def local_feature_matching(self):
        """
        Match local features between query and database images.
        """
        
        self.path_to_local_features = extract_features.main(
            conf=self.local_feature_conf,
            image_dir=self.path_to_images,
            export_dir=self.path_to_outputs,
        )    

        self.path_to_local_matches = match_features.main(
            conf=self.local_match_conf,
            pairs=self.path_to_retrieval_pairs,
            features=self.local_feature_conf['output'],
            export_dir=self.path_to_outputs,
        )


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
        with open(self.path_to_retrieval_pairs, 'r') as f:
            db_names = [line.split()[1] for line in f if line.startswith(query_name)]
        return db_names

    def visualize_local_matches(self, query_name: str, db_names: List[str] = None, min_score: int = 0, db_limit: int = 10) -> None:
        '''
        Visualize local matches between query image and paired database images.
        Draw points and lines between matched keypoints, with query image on the left, and database image on the right.
        '''

        if not query_name.startswith(self.query_prefix):
            query_name = self.query_prefix + '/' + query_name
        query_image = read_image(self.path_to_images / query_name)

        if db_names:
            self.check_retrieval_pairs(query_name, db_names)  
        else:
            db_names = self.get_retrieved_db_images(query_name)

        db_names = db_names[:db_limit]

        for db_name in db_names:

            if not db_name.startswith(self.db_prefix):
                db_name = self.db_prefix + '/' + db_name
            db_image = read_image(self.path_to_images / db_name)
            # pair, reverse = find_pair(local_matches, query_name, db_name)
            # if pair:
            matches, scores = get_matches(self.path_to_local_matches, query_name, db_name)

            if self.path_to_local_features == None:
                raise ValueError("Local features not yet extracted. Run local_feature_matching() first.")
            query_keypoints = get_keypoints(self.path_to_local_features, query_name)
            db_keypoints = get_keypoints(self.path_to_local_features, db_name)

            # Visualize matches
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            ax[0].imshow(query_image)
            ax[1].imshow(db_image)
            for i, (q, db) in enumerate(matches):
                if scores[i] < min_score:
                    continue
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