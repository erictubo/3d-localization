from pathlib import Path


path_to_sfm_data = Path('/Users/eric/Downloads/notre_dame_front_facade/dense/sparse/')

path_to_evaluation = Path('/Users/eric/Downloads/evaluation/')


notre_dame = {
    'sfm_model': '/Users/eric/Downloads/notre_dame_front_facade/dense/sparse/',
    'cad_models': {
        'notre dame B' : {
            'prefix' : 'notre_dame_B/',
        },
        'notre dame E' : {
            'prefix' : 'notre_dame_E/',
        }
    }
}

reference_models = [notre_dame]