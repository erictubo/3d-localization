import os

user = os.getlogin()

data_dir = '/home/'+user+'/Documents/data'

evaluation_dir = data_dir + '/evaluation'
models_dir = data_dir + '/models'
renders_dir = data_dir + '/renders'
