from pathlib import Path


path_to_3d_models = Path('/Users/eric/Documents/Studies/MSc Robotics/Thesis/3D Models/')

path_to_evaluation = Path('/Users/eric/Documents/Studies/MSc Robotics/Thesis/Evaluation/')


notre_dame = {
    'sfm_model': path_to_3d_models / 'Notre Dame/Reference/dense/sparse/',
    'cad_models': {
        'notre dame B' : {
            'prefix' : 'notre_dame_B/',
        },
        'notre dame E' : {
            'prefix' : 'notre_dame_E/',
        }
    }
}

reichstag = {
    'sfm_model': path_to_3d_models / 'Reichstag/Reference/dense/sparse/',
    'cad_models': {
        'reichstag B' : {
            'prefix' : 'reichstag_B/',
        },
        'reichstag E' : {
            'prefix' : 'reichstag_E/',
        }
    }
}

reference_models = [notre_dame]