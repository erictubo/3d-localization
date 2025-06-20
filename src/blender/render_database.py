"""
Blender batch rendering script for automatic database view generation.
Renders multiple views of CAD models using different camera settings and saves outputs.
"""
import sys
import os

script_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_directory)

from renderer import Renderer
from data import models_dir, renders_dir

models = {
    'Pantheon': ['B']
    # 'Notre Dame': ['B', 'E'],
    # 'Reichstag': ['A', 'B'],
}


for name in models:
    for id in models[name]:

        print(f'Model: {name} / {id}')
        full_name = (name + ' ' + id) # e.g. Notre Dame B

        blend_file = f'{models_dir}/{name}/{id}/{full_name}.blend'
        assert os.path.exists(blend_file), f"Blend file not found: {blend_file}"

        render_dir = f'{renders_dir}/{full_name}/'
        
        renderer = Renderer(
            blend_file,
            render_dir,
            )
        
        # for f, d in [(30, 110), (60, 220), (120, 440)]:
        #     renderer.render_ground_views(
        #         distances=[d],
        #         h_steps = 72,
        #         heights = [2, 20, 50],
        #         focal_lengths=[f],
        #         )


        '''
        Camera settings
        '''

        # Wide angle from minimum distance
        d_wide = 110
        f_wide = 25

        # Wide angle from 2x distance
        d_normal = 220
        f_normal = 75

        # Telephoto angle from 4x distance
        d_tele = 440
        f_tele = 150

        # Wide angle from minimum distance for detailed views
        d_detail = 110
        f_detail = 50

        v_offsets = [-10, 0, 10]
        h_offsets = [-10, 0, 10]
        
        if input('Render all or test? [all/test]') == 'all':
            h_steps = 72
            heights = [2, 20, 50]
        else:
            h_steps = 1
            heights = [20]


        # Wide from 110m without offsets
        # 1d x 72h x 3z x 1v x 1h = 216 renders
        renderer.render_ground_views(
            distances=[d_wide],
            h_steps = h_steps,
            heights = heights,
            focal_lengths=[f_wide],
            v_offsets_deg=[0],
            h_offsets_deg=[0],
        )

        # Normal from 220m without offsets
        # 1d x 72h x 3z x 1v x 1h = 216 
        renderer.render_ground_views(
            distances=[d_normal],
            h_steps = h_steps,
            heights = heights,
            focal_lengths=[f_normal],
            v_offsets_deg=[0],
            h_offsets_deg=[0]
        )

        # Telephoto from 440m without offsets
        # 1d x 72h x 3z x 1v x 1h = 216 renders
        renderer.render_ground_views(
            distances=[d_tele],
            h_steps = h_steps,
            heights = heights,
            focal_lengths=[f_tele],
            v_offsets_deg=[0],
            h_offsets_deg=[0]
        )

        # Normal from 110m with offsets for detailed views
        # 1d x 72h x 3z x 3v x 3h = 1944 renders
        renderer.render_ground_views(
            distances=[d_detail],
            h_steps = h_steps,
            heights = heights,
            focal_lengths=[f_detail],
            v_offsets_deg=v_offsets,
            h_offsets_deg=h_offsets,
        )