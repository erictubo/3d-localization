"""
Blender data paths setup.
Defines user-specific data, evaluation, models, and renders directories for Blender scripts.
"""
import os

user = os.getlogin()

data_dir = '/home/'+user+'/Documents/data'

evaluation_dir = data_dir + '/evaluation'
models_dir = data_dir + '/models'
renders_dir = data_dir + '/renders'
