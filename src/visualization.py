import numpy as np
from typing import Tuple
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt


class Visualization:

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

        query_names = [f.name for f in path_to_query_images.glob('*') if f.suffix in ['.jpg', '.png', '.jpeg']]
        render_names = [f.name for f in path_to_render_images.glob('*') if f.suffix in ['.jpg', '.png', '.jpeg']]

        for query_name in query_names:

            render_name = next((name for name in render_names if 'query_' + query_name.split('.')[0] == name.split('.')[0]), None)

            if render_name:
                render_names.remove(render_name)
            else:
                raise ValueError(f'No render image found for query image {query_name}')
            
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

    @staticmethod
    def visualize_depth_map(
            path_to_depth: Path,
            name: str,
            path_to_output: Path = None,
        ):
        """
        Visualize npy/npz depth map with colors according to depth values and a legend.
        Goal: to check if depth map is correct.
        """

        name = name.split('.')[0]
        # if not name.endswith('_depth'):
        #     name += '_depth'
        depth_name = name + '.npz'

        depth_map = np.load(path_to_depth / depth_name)['depth']


        cmap = plt.get_cmap('viridis')
        cmap.set_bad(color='white')
        
        # Create a masked array, masking zero values
        masked_depth_map = np.ma.masked_where(depth_map == 0, depth_map)
        
        fig, ax = plt.subplots(figsize=(10, 8))

        # Create a color-coded image of the depth map
        im = ax.imshow(masked_depth_map, cmap=cmap)

        # Reverse the y-axis
        # ax.invert_yaxis()
        
        # Add a colorbar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Depth', rotation=270, labelpad=15)
        
        # ax.set_title('Depth Map Visualization')

        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.axis('off')

        ax.set_xticks([])
        ax.set_yticks([])
        
        if not path_to_output:
            plt.show()
        else:
            plt.savefig(path_to_output / f'{name}.png', transparent=True)
        plt.close()

    @staticmethod
    def visualize_scene_coordinate_map(
            path_to_scene_coordinates: Path,
            name: str,
            path_to_output: Path = None,
            format: str = 'npz',
        ):
        """
        Visualize scene coordinates.
        """

        name = name.split('.')[0]
        if format == 'dat':
            import torch
            coordinates_name = name + '.dat'
            coordinates = torch.load(path_to_scene_coordinates / coordinates_name)

            # Format:(3, H, W)
            # Convert to numpy array of shape (H, W, 3)
            coordinates = coordinates.permute(1, 2, 0).numpy()
        else:
            coordinates_name = name + '.npz'
            coordinates = np.load(path_to_scene_coordinates / coordinates_name)['scene_coordinates']


        print(coordinates.shape)

        cmap = plt.get_cmap('viridis')

        mask = np.all(coordinates == [0., 0., 0.], axis=-1)
        masked_coordinates = np.ma.masked_array(coordinates, mask=np.repeat(mask[:, :, np.newaxis], 3, axis=2))
        
        # Normalize the coordinates to the range [0, 1]
        normalized_coordinates = (masked_coordinates - masked_coordinates.min(axis=(0, 1))) / (masked_coordinates.max(axis=(0, 1)) - masked_coordinates.min(axis=(0, 1)))

        # set all masked values to white
        normalized_coordinates = np.where(mask[:, :, np.newaxis], 1, normalized_coordinates)
        
        fig, ax = plt.subplots(figsize=(10, 8))

        # Use the normalized coordinates as RGB values
        im = ax.imshow(normalized_coordinates, cmap=cmap)
        
        # ax.set_title('3D Coordinate Visualization')

        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.axis('off')

        ax.set_xticks([])
        ax.set_yticks([])
        
        if not path_to_output:
            plt.show()
        else:
            plt.savefig(path_to_output / f'{name}_coordinates.png', transparent=True)
        plt.close()


if __name__ == '__main__':

    # path_to_query_images = Path('/Users/eric/Documents/Studies/MSc Robotics/Thesis/Evaluation/notre_dame_B/inputs/query/images')
    # path_to_render_images = Path('/Users/eric/Documents/Studies/MSc Robotics/Thesis/Evaluation/notre_dame_B/outputs/meshloc_out/patch2pix/renders/images')
    # path_to_overlays = Path('/Users/eric/Documents/Studies/MSc Robotics/Thesis/Evaluation/notre_dame_B/outputs/meshloc_out/patch2pix/overlays')

    # Visualization.overlay_query_and_rendered_images(path_to_query_images, path_to_render_images, path_to_overlays)

    path_to_scene_coordinates = Path('/Users/eric/Documents/Studies/MSc Robotics/Thesis/Data/Evaluation/notre dame B/inputs/database/scene coordinates/')

    names = ['f30_d110_z20_h195.npz'] #, 'f30_d110_z20_h210.npz', 'f30_d110_z20_h225.npz', 'f30_d110_z20_h240.npz', 'f30_d110_z20_h255.npz']

    for name in names:
        Visualization.visualize_scene_coordinate_map(path_to_scene_coordinates, name)

    path_to_scene_coordinates = Path('/Users/eric/Documents/Studies/MSc Robotics/Thesis/Data/GLACE/notre dame (SFM)/train/init/')
    format = 'dat'

    names = ['49379137_4824496602', '49452387_8136855930', '49610648_2419143510']

    for name in names:
        Visualization.visualize_scene_coordinate_map(path_to_scene_coordinates, name, format=format, path_to_output=path_to_scene_coordinates)
