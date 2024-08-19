from pathlib import Path


path_to_3d_models = Path('/Users/eric/Documents/Studies/MSc Robotics/Thesis/3D Models/')
path_to_evaluation = Path('/Users/eric/Documents/Studies/MSc Robotics/Thesis/Evaluation/')

path_to_blender = Path('/Users/eric/Library/Mobile Documents/com~apple~CloudDocs/Blender/')


class Model:

    def __init__(self, name: str):
        self.name = name

        self.path_to_reference = path_to_3d_models / name / 'Reference/dense/sparse/'


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

    def __init__(self, model: Model, id: str, target_name: str = None):
        self.model = model
        self.id = id.upper()
        self.prefix = self.model.name.replace(' ', '_').lower() + f'_{id}'        
        self.get_paths(self.prefix)
        
        if target_name: 
            blend_name = self.prefix.replace('_', ' ')
            self.blend_file = path_to_blender / f'assets/models/{blend_name}/{blend_name}.blend'
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
    
    notre_dame = Model('Notre Dame')

    notre_dame_B = CadModel(notre_dame, 'B', 'SketchUp')