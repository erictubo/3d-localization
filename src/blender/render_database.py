import sys
import os

script_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_directory)

from render import Blender

model_dir = '/Users/eric/Documents/Studies/MSc Robotics/Thesis/3D Models'
renders_dir = '/Users/eric/Library/Mobile Documents/com~apple~CloudDocs/Blender/renders'

# notre dame B: {model_dir}/Notre Dame/B/notre dame B.blend
# notre dame E
# reichstag A
# reichstag B
# reichstag E
# st peters square B:  /St Peters Square/B/st peters square B.blend
# st peters square C: /St Peters Square/C/st peters square C.blend

models = {
    'Notre Dame': ['B'], #, 'E'],
    'Reichstag': ['A', 'B'], #, 'E'],
    # 'St Peters Square': ['B', 'C']
}


for name in models:
    for id in models[name]:

        print(f'Model: {name} / {id}')
        full_name = (name.lower() + ' ' + id) # e.g. notre dame B

        blend_file = f'{model_dir}/{name}/{id}/{full_name}.blend'
        assert os.path.exists(blend_file), f"Blend file not found: {blend_file}"

        render_dir = f'{renders_dir}/{full_name}/'
        
        blender = Blender(
            blend_file,
            render_dir,
            )
        
        for f, d in [(30, 110), (60, 220), (120, 440)]:
            blender.render_ground_views(
                distances=[d],
                h_steps = 24,
                heights = [2, 20],
                focal_lengths=[f],
                )
