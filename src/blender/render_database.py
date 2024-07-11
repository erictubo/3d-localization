
import sys
sys.path.append('/Users/eric/Developer/model_localization/blender')

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
        
        blender.render_ground_views(
            distances=[110, 160],
            h_steps = 72,
            heights = [2, 20]
            )        
