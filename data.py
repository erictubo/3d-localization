from pathlib import Path


path_to_sfm_data = Path('/Users/eric/Downloads/notre_dame_front_facade/dense/sparse/')

path_to_evaluation = Path('/Users/eric/Downloads/evaluation/')

models = {
    'notre dame front facade': {
        'path_to_sfm_model': Path('/Users/eric/Downloads/notre_dame_front_facade/dense/sparse/'),
        'cad_models': {
            'notre dame B',
            'notre dame E',
        }
    },
}

