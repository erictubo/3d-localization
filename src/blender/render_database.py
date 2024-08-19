import sys
import os

script_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_directory)


from render import Blender
from render_data import blender_dir, models


if __name__ == "__main__":

    for model in ['notre dame B']:

        blend_file = models[model]['blend_file']
        target_name = models[model]['target_name']

        render_dir = f'{blender_dir}renders/{model}/'

        blender = Blender(
            blend_file,
            render_dir,
            target_name,
            )

        # blender.render_orbit_views(
        #     h_steps = 36,
        #     v_angles_deg = [-10, 0, 10, 20]
        #     )
        
        # blender.render_ground_views(
        #     distances=[110, 160],
        #     h_steps = 72,
        #     heights = [2, 20]
        #     )

        # blender.render_ground_views(
        #     distances=[110, 160, 210],
        #     h_steps = 8,
        #     heights = [2, 20],
        #     focal_lengths=[35, 50, 70]
        #     )



        # MULTIPLE FOCAL LENGTHS

        for f, d in [(30, 110), (60, 220), (120, 440)]:
            blender.render_ground_views(
                distances=[d],
                h_steps = 24,
                heights = [2, 20],
                focal_lengths=[f]
                )
