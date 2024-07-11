import numpy as np
from typing import Tuple
from pathlib import Path


class Evaluation:

    def __init__(self):
        pass

    @staticmethod
    def overlay_query_and_rendered_images(
            path_to_query_images: Path,
            path_to_render_images: Path,
            path_to_overlays: Path):
        """
        Overlay query and rendered images in 2x2 grid.
        """
        assert path_to_query_images.exists(), path_to_query_images
        assert path_to_render_images.exists(), path_to_render_images
        
        query_names = [f.name for f in path_to_query_images.glob('*')]
        render_names = [f.name for f in path_to_render_images.glob('*')]

        # print('Query names: ', query_names)
        # print('Render names: ', render_names)

        for query_name in query_names:

            from PIL import Image

            render_name = next((name for name in render_names if query_name.replace('.jpg','') in name), None)
            
            if render_name:
                # Load the two images
                query_image = Image.open(path_to_query_images / query_name)
                render_image = Image.open(path_to_render_images / render_name)

                # Resize the images while maintaining aspect ratio
                # check that query image and render image have the same dimensions
                if query_image.size != render_image.size:
                    width, height = 600, 600  # Desired size
                else:
                    width, height = query_image.size
                query_image = query_image.resize((width, height), Image.Resampling.LANCZOS)
                render_image = render_image.resize((width, height), Image.Resampling.LANCZOS)

                # Create a new blank image with combined dimensions
                combined_width = width
                combined_height = height
                combined_image = Image.new('RGB', (combined_width, combined_height))

                # Paste the cropped images onto the combined image
                combined_image.paste(query_image.crop((0, 0, width//2, height//2)), (0, 0))  # Top left quarter
                combined_image.paste(query_image.crop((width//2, height//2, width, height)), (width//2, height//2))  # Bottom right quarter
                combined_image.paste(render_image.crop((0, height//2, width//2, height)), (0, height//2))  # Top right quarter
                combined_image.paste(render_image.crop((width//2, 0, width, height//2)), (width//2, 0))  # Bottom left quarter

                # Save the combined image
                if not path_to_overlays.exists():
                    path_to_overlays.mkdir()
                combined_image.save(path_to_overlays / query_name)