from pathlib import Path


path_to_3d_models = Path('/Users/eric/Documents/Studies/MSc Robotics/Thesis/3D Models/')

path_to_evaluation = Path('/Users/eric/Documents/Studies/MSc Robotics/Thesis/Evaluation/')


notre_dame = {
    'sfm model': path_to_3d_models / 'Notre Dame/Reference/dense/sparse/',
    'cad models': {
        'notre dame B' : {
            'prefix' : 'notre_dame_B/',
        },
        'notre dame E' : {
            'prefix' : 'notre_dame_E/',
        }
    }
}

reichstag = {
    'sfm model': path_to_3d_models / 'Reichstag/Reference/dense/sparse/',
    'cad models': {
        'reichstag B' : {
            'prefix' : 'reichstag_B/',
        },
        'reichstag E' : {
            'prefix' : 'reichstag_E/',
        }
    }
}

models = [notre_dame]