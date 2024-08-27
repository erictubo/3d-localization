import sys
import os

script_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_directory)

from render import Blender
from render_data import models_dir, renders_dir

models = {
    'Notre Dame': ['B'], #, 'E'],
    # 'Reichstag': ['A', 'B'], #, 'E'],
    # 'St Peters Square': ['B', 'C']
}


for name in models:
    for id in models[name]:

        print(f'Model: {name} / {id}')
        full_name = (name.lower() + ' ' + id) # e.g. notre dame B

        blend_file = f'{models_dir}/{name}/{id}/{full_name}.blend'
        assert os.path.exists(blend_file), f"Blend file not found: {blend_file}"

        render_dir = f'{renders_dir}/{full_name}/'
        
        blender = Blender(
            blend_file,
            render_dir,
            )
        
        for f, d in [(30, 110), (60, 220), (120, 440)]:
            blender.render_ground_views(
                distances=[d],
                h_steps = 72,
                heights = [2, 20, 50],
                focal_lengths=[f],
                )
