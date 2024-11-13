import os
from pathlib import Path

"""
Handling of data paths for each reference model and the corresponding CAD models.

Simply set path_to_data and structure the data inside it as follows:

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
"""

path_to_data = Path('/Users/eric/Documents/Studies/MSc Robotics/Thesis/Data/')

path_to_models = path_to_data / '3D Models/'
path_to_evaluation = path_to_data / 'Evaluation/'


class Model:

    def __init__(self, name: str):
        self.name = name.title()
        print(f'Model: {self.name}')

        self.path_to_reference_model = path_to_models / name / 'Reference/dense/sparse/'
        self.path_to_reference_images = path_to_models / name / 'Reference/dense/images/'


class CadModel:

    """
    Dataset file structure:
    - ground truth
        - 
    - inputs
        - database
        - query
    - outputs
        - features
        - meshloc_match
        - meshloc_out
    """

    def __init__(self, model: Model, id: str, target_name: str = 'Model'):
        self.model = model
        self.id = id.upper()
        print(f'CAD Model: {self.model.name} {self.id}')
        self.prefix = f'{self.model.name.lower()} {id}' # e.g. notre dame B
        
        self.blend_file = path_to_models / model.name / self.id / f'{self.prefix}.blend'
        assert os.path.exists(self.blend_file), f"Blend file not found: {self.blend_file}"

        self.get_paths(self.prefix)
        
        self.target_name = target_name

    def get_paths(self, prefix: str):
        self.path_to_dataset = path_to_evaluation / f'{prefix}/'
        self.path_to_ground_truth = self.path_to_dataset / 'ground truth/'
        self.path_to_inputs = self.path_to_dataset / 'inputs/'
        self.path_to_outputs = self.path_to_dataset / 'outputs/'

        assert self.path_to_ground_truth.exists(), \
            f"Ground truth directory not found: {self.path_to_ground_truth}"
        assert self.path_to_inputs.exists(), \
            f"Inputs directory not found: {self.path_to_inputs}"
        if not self.path_to_outputs.exists():
            self.path_to_outputs.mkdir()

        self.path_to_query = self.path_to_inputs / 'query/'
        self.path_to_query_images = self.path_to_query / 'images/'
        self.path_to_database = self.path_to_inputs / 'database/'

        self.path_to_features = self.path_to_outputs / 'features/'
        self.path_to_meshloc_out = self.path_to_outputs / 'meshloc_out/'

        self.path_to_retrieval_pairs = self.path_to_features / "pairs-from-retrieval.txt"
        self.path_to_retrieval_pairs_meshloc = self.path_to_features / "pairs-from-retrieval-meshloc.txt"


if __name__ == '__main__':
    
    notre_dame = Model('notre dame')
    notre_dame_B = CadModel(notre_dame, 'b')