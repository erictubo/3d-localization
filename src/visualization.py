import numpy as np
from typing import Tuple
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt


class Visualization:

    def __init__(self):
        pass

    @staticmethod
    def overlay_query_and_rendered_image(
            query_image: Image,
            render_image: Image,
            subsample: int = 1,
            tiles: Tuple[int, int] = None,
            tiles_pix: Tuple[int, int] = (160, 160),
        ):

        # check that subsample is a power of 2
        assert subsample & (subsample - 1) == 0, 'Subsample must be a power of 2'

        # Resize the images while maintaining aspect ratio
        # check that query image and render image have the same dimensions
        assert query_image.size == render_image.size, \
            f'Query image size {query_image.size} and render image size {render_image.size} do not match'
        
        init_width, init_height = query_image.size

        # resize to 960x640 (depends on the aspect ratio)
        if init_width > init_height:
            width, height = 960, 640
        else:
            width, height = 640, 960
        
        width, height = width // subsample, height // subsample
                    
        query_image = query_image.resize((width, height), Image.Resampling.LANCZOS)
        render_image = render_image.resize((width, height), Image.Resampling.LANCZOS)

        print(f'Query image size: {query_image.size}')
        print(f'Render image size: {render_image.size}')

        if tiles:
            grid_width, grid_height = tiles
            cell_width, cell_height = width // grid_width, height // grid_height
        elif tiles_pix:
            cell_width, cell_height = (tiles_pix[0] // subsample, tiles_pix[1] // subsample)
            grid_width, grid_height = width // cell_width, height // cell_height
        else:
            raise ValueError('Either tiles or tiles_pix must be specified')
        
        print(f'Grid size: {grid_width}x{grid_height}')
        print(f'Cell size: {cell_width}x{cell_height}')

        # Convert images to numpy arrays for easier manipulation
        query_array = np.array(query_image)
        render_array = np.array(render_image)

        # Subsample the arrays
        query_array = query_array[:, :, :3]
        render_array = render_array[:, :, :3]

        # Create a new blank image for the grid
        combined_image = Image.new('RGB', (width, height))
        combined_array = np.array(combined_image)


        for i in range(grid_height):
            for j in range(grid_width):
                cell_slice = slice(i*cell_height, (i+1)*cell_height), slice(j*cell_width, (j+1)*cell_width)
                
                if (i + j) % 2 == 0:  # Even cells get query image
                    combined_array[cell_slice] = query_array[cell_slice]
                else:  # Odd cells get render image
                    combined_array[cell_slice] = render_array[cell_slice]

        # Convert back to PIL Image
        combined_image = Image.fromarray(combined_array)
        
        return combined_image

    @staticmethod
    def overlay_query_and_rendered_images(
            path_to_query_images: Path,
            path_to_render_images: Path,
            path_to_overlays: Path = None,
            subsample: int = 1,
            tiles: Tuple[int, int] = None,
            tiles_pix: Tuple[int, int] = (160, 160),
        ):
        """
        Overlay query and rendered images in NxM grid.
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

                combined_image = Visualization.overlay_query_and_rendered_image(
                    query_image,
                    render_image,
                    subsample=subsample,
                    tiles=tiles, tiles_pix=tiles_pix,
                )

                # Save the combined image
                if path_to_overlays:
                    if not path_to_overlays.exists():
                        path_to_overlays.mkdir()
                    combined_image.save(path_to_overlays / query_name)
                else:
                    combined_image.show()

    @staticmethod
    def visualize_depth_map(
            path_to_depth: Path,
            name: str,
            path_to_output: Path = None,
            output_name: str = None,
            extension: str = 'npz',
            mm_to_m: bool = False,
            depth_range: Tuple[int, int] = None,
        ):
        """
        Visualize npy/npz depth map with colors according to depth values and a legend.
        Goal: to check if depth map is correct.
        """

        name = name.split('.')[0]
        # if not name.endswith('_depth'):
        #     name += '_depth'
        depth_name = name + '.' + extension.lower()

        if extension == 'npz':
            depth_map = np.load(path_to_depth / depth_name)['depth']
        elif extension == 'npy':
            depth_map = np.load(path_to_depth / depth_name)

        cmap = plt.get_cmap('viridis')
        cmap.set_bad(color='white')
        
        # Create a masked array, masking zero values
        masked_depth_map = np.ma.masked_where(depth_map == 0, depth_map)

        if mm_to_m:
            masked_depth_map = masked_depth_map / 1000
        
        fig, ax = plt.subplots(figsize=(10, 8))

        # Set the range of the color map
        if depth_range:
            assert depth_range[0] >= 0 and depth_range[1] > depth_range[0], 'Invalid depth range'
            # check that the depth map values are within the specified range
            assert masked_depth_map.min() >= depth_range[0] and masked_depth_map.max() <= depth_range[1], \
                f'Depth map values {np.floor(masked_depth_map.min())} and {np.ceil(masked_depth_map.max())} are not within the specified range'
            im = ax.imshow(masked_depth_map, cmap=cmap, vmin=depth_range[0], vmax=depth_range[1])
        else:
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
            if not output_name:
                output_name = f'{name}_depth'
            plt.savefig(path_to_output / f'{output_name}.png', transparent=True)
        plt.close()

    @staticmethod
    def visualize_scene_coordinate_map(
            path_to_scene_coordinates: Path,
            name: str,
            path_to_output: Path = None,
            output_name: str = None,
            format: str = 'npz',
            x_range: Tuple[int, int] = None, # (min, max) for color of X coordinate
            y_range: Tuple[int, int] = None, # (min, max) for color of Y coordinate
            z_range: Tuple[int, int] = None, # (min, max) for color of Z coordinate
            ignore_limit: float = 0.05,
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

        cmap = plt.get_cmap('viridis')

        mask = np.all(coordinates == [0., 0., 0.], axis=-1)
        masked_coordinates = np.ma.masked_array(coordinates, mask=np.repeat(mask[:, :, np.newaxis], 3, axis=2))
        
        min_coords = np.floor(masked_coordinates.min(axis=(0, 1)))
        max_coords = np.ceil(masked_coordinates.max(axis=(0, 1)))

        # print(f'Min coordinates: {min_coords}')
        # print(f'Max coordinates: {max_coords}')

        if (x_range and y_range and z_range):

            min_coords_limit = np.array([x_range[0], y_range[0], z_range[0]])
            max_coords_limit = np.array([x_range[1], y_range[1], z_range[1]])


            # calculate the quantity of any coordinates outside the specified ranges (any dimension)
            num_coords_outside = np.sum(
                np.any((masked_coordinates < min_coords_limit) | (masked_coordinates > max_coords_limit), axis=-1)
            )

            print(f'Number of coordinates outside the specified ranges: {num_coords_outside}')

            # if less than 5% of the coordinates are outside the specified ranges, use the specified ranges
            # remove the coordinates outside the specified ranges
            if num_coords_outside < ignore_limit * masked_coordinates.size:
                masked_coordinates = np.clip(masked_coordinates, min_coords_limit, max_coords_limit)
            else:
                raise ValueError(f'Percentage of coordinates outside the specified ranges {num_coords_outside / masked_coordinates.size} is greater than the limit {ignore_limit}')


            # assert (min_coords >= min_coords_limit).all() and (max_coords <= max_coords_limit).all(), \
            #     f'min_coords {min_coords} and max_coords {max_coords} are not within the specified ranges'


            # Normalize the coordinates such that [min, max] -> [0, 1]
            normalized_coordinates = (masked_coordinates - min_coords_limit) / (max_coords_limit - min_coords_limit)

        else:
            # Normalize the coordinates to the range [0, 1]
            normalized_coordinates = (masked_coordinates - min_coords) / (max_coords - min_coords)


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
            if not output_name:
                output_name = f'{name}_coordinates'
            plt.savefig(path_to_output / f'{output_name}.png', transparent=True)
        plt.close()


    def compare_depth_maps(
            path_to_depth_1: Path,
            path_to_depth_2: Path,
            name: str,
            path_to_output: Path = None,
            mm_to_m: bool = False,
    ):
        """
        Visualize the pixel-wise difference between two depth maps.
        Difference = absolute difference between the two depth values
        """

        def load_depth_map(path_to_depth: Path):
            format = path_to_depth.suffix[1:]
            if format == 'npz':
                depth_map = np.load(path_to_depth)['depth']
            elif format == 'npy':
                depth_map = np.load(path_to_depth)
            mask = depth_map == 0.
            return depth_map, mask

        depth_map_1, mask_1 = load_depth_map(path_to_depth_1)
        depth_map_2, mask_2 = load_depth_map(path_to_depth_2)

        # if the two depth maps have different sizes, resize the smaller one to the size of the larger one
        # by repeating the values
        
        if depth_map_1.shape != depth_map_2.shape:
            
            large_depth_map, small_depth_map = (depth_map_1, depth_map_2) if depth_map_1.size > depth_map_2.size else (depth_map_2, depth_map_1)
            large_mask, small_mask = (mask_1, mask_2) if depth_map_1.size > depth_map_2.size else (mask_2, mask_1)

            scale_x = large_depth_map.shape[1] / small_depth_map.shape[1]
            scale_y = large_depth_map.shape[0] / small_depth_map.shape[0]

            resized_depth_map_new = np.zeros_like(large_depth_map)
            resized_mask_new = np.zeros_like(large_mask)
            
            for y in range(resized_depth_map_new.shape[0]):
                for x in range(resized_depth_map_new.shape[1]):
                    
                    x_ = int( (x+0.5) / scale_x)
                    y_ = int( (y+0.5) / scale_y)

                    resized_depth_map_new[y, x] = small_depth_map[y_, x_]
                    resized_mask_new[y, x] = small_mask[y_, x_]

            depth_map_1, mask_1 = (large_depth_map, large_mask)
            depth_map_2, mask_2 = (resized_depth_map_new, resized_mask_new)


            # Option 2: subsample the larger depth map

            # resized_depth_map = np.zeros_like(small_depth_map)
            # resized_mask = np.zeros_like(small_mask)

            # for y in range(resized_depth_map.shape[0]):
            #     for x in range(resized_depth_map.shape[1]):
            #         x_ = int( (x+0.5) * scale_x - 0.5)
            #         y_ = int( (y+0.5) * scale_y - 0.5)

            #         resized_depth_map[y, x] = large_depth_map[y_, x_]
            #         resized_mask[y, x] = large_mask[y_, x_]

            # depth_map_1, mask_1 = (small_depth_map, small_mask)
            # depth_map_2, mask_2 = (resized_depth_map, resized_mask)

            
            assert depth_map_1.shape == depth_map_2.shape, \
                f'Depth maps {path_to_depth_1.name} and {path_to_depth_2.name} have different sizes after resizing: {depth_map_1.shape} and {depth_map_2.shape}'

        mask = mask_1 | mask_2

        # Calculate the difference
        difference = np.abs(depth_map_1 - depth_map_2)
        masked_difference = np.ma.masked_array(difference, mask=mask)

        if mm_to_m:
            masked_difference = masked_difference / 1000

        # use a color map where low values are green and high values are red
        cmap = plt.get_cmap('Spectral').reversed()
        cmap.set_bad(color='white')
        
        fig, ax = plt.subplots(figsize=(10, 8))

        # Create a color-coded image of the depth map
        im = ax.imshow(masked_difference, cmap=cmap, vmin=0)

        # Reverse the y-axis
        # ax.invert_yaxis()
        
        # Add a colorbar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Difference', rotation=270, labelpad=15)
        
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


    def compare_scene_coordinate_maps(
            path_to_scene_coordinates_1: Path,
            path_to_scene_coordinates_2: Path,
            name: str,
            path_to_output: Path = None,
        ):
        """
        Visualize the pixel-wise difference between two scene coordinate maps.
        Difference = distance between the two points in 3D space
        """
        
        def load_coordinates(path_to_scene_coordinates: Path):
            format = path_to_scene_coordinates.suffix[1:]
            if format == 'dat':
                import torch
                coordinates = torch.load(path_to_scene_coordinates)
                # Torch tensor (3, H, W) -> numpy array (H, W, 3)
                coordinates = coordinates.permute(1, 2, 0).numpy()
            else:
                coordinates = np.load(path_to_scene_coordinates)['scene_coordinates']
            mask = np.all(coordinates == [0., 0., 0.], axis=-1)
            return coordinates, mask
        
        coordinates_1, mask_1 = load_coordinates(path_to_scene_coordinates_1)
        coordinates_2, mask_2 = load_coordinates(path_to_scene_coordinates_2)

        # combine the masks: multiply
        mask = mask_1 | mask_2

        # Calculate the difference
        difference = np.linalg.norm(coordinates_1 - coordinates_2, axis=-1)
        masked_difference = np.ma.masked_array(difference, mask=mask)

        cmap = plt.get_cmap('Spectral').reversed()
        cmap.set_bad(color='white')
        
        fig, ax = plt.subplots(figsize=(10, 8))

        # Create a color-coded image of the depth map
        im = ax.imshow(masked_difference, cmap=cmap)

        # Reverse the y-axis
        # ax.invert_yaxis()
        
        # Add a colorbar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Distance', rotation=270, labelpad=15)
        
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

        return np.mean(masked_difference)


if __name__ == '__main__':

    from data import path_to_data

    # path_to_query_images = path_to_data / 'Evaluation/notre dame B/inputs/query/images'
    # path_to_render_images = path_to_data / 'Evaluation/notre dame B/ground truth/renders/images'
    # path_to_overlays = path_to_data / 'Evaluation/notre dame B/ground truth/overlays'
    # path_to_overlays_subsampled = path_to_data / 'Evaluation/notre dame B/ground truth/overlays/subsampled'

    # Visualization.overlay_query_and_rendered_images(path_to_query_images, path_to_render_images, path_to_overlays, subsample=1)
    # Visualization.overlay_query_and_rendered_images(path_to_query_images, path_to_render_images, path_to_overlays_subsampled, subsample=8)
    # Visualization.overlay_query_and_rendered_images(path_to_query_images, path_to_render_images, subsample=8)



    # path_to_scene_coordinates = path_to_data / 'Evaluation/notre dame B/inputs/database/scene coordinates/'

    # names = ['f30_d110_z20_h195.npz'] #, 'f30_d110_z20_h210.npz', 'f30_d110_z20_h225.npz', 'f30_d110_z20_h240.npz', 'f30_d110_z20_h255.npz']

    # for name in names:
    #     Visualization.visualize_scene_coordinate_map(path_to_scene_coordinates, name)



    path_to_reconstructed_coords = path_to_data / 'GLACE/pantheon (SFM)/test/init/'
    path_to_reconstructed_depth = path_to_data / 'GLACE/pantheon (SFM)/test/depth/'
    path_to_rendered_depth = path_to_data / 'GLACE/pantheon (renders)/test/depth/'
    format = 'dat'

    names = ['00318896_2265892479', '00488011_10505838106', '00617912_545466582', '00862207_425411470']

    path_to_output = Path('/Users/eric/Downloads') # None

    depth_range = (0, 240)

    # Min coordinates: [-31.0 -28.0 3.0]
    # Max coordinates: [111.0 85.0 46.0]
    # Min coordinates: [-142.0 -109.0 2.0]
    # Max coordinates: [43.0 73.0 42.0]
    # Min coordinates: [-14.0 22.0 11.0]
    # Max coordinates: [17.0 40.0 33.0]
    # Min coordinates: [-40.0 10.0 2.0]
    # Max coordinates: [18.0 41.0 33.0]

    # Combined: [-142.0 -109.0 2.0] [111.0 85.0 46.0]

    coords_range = (-142, 111), (-109, 85), (2, 46)


    for i, name in enumerate(names):
        Visualization.visualize_depth_map(
            path_to_depth=path_to_reconstructed_depth,
            mm_to_m=True,
            name=name,
            extension='npy',
            path_to_output=path_to_output,
            output_name=f'{i+1}_depth_reconstructed',
            depth_range=depth_range,
        )
        Visualization.visualize_scene_coordinate_map(
            path_to_scene_coordinates=path_to_reconstructed_coords,
            name=name,
            format=format,
            path_to_output=path_to_output,
            output_name=f'{i+1}_coordinates',
            x_range=coords_range[0],
            y_range=coords_range[1],
            z_range=coords_range[2],
            # x_range=(-120, 100),
            # y_range=(-80, 80),
            # z_range=(-20, 120),
        )

        Visualization.visualize_depth_map(
            path_to_depth=path_to_rendered_depth,
            mm_to_m=True,
            name=f'query_{name}',
            extension='npy',
            path_to_output=path_to_output,
            output_name=f'{i+1}_depth_rendered',
            depth_range=depth_range,
        )

        
        Visualization.compare_depth_maps(
                path_to_depth_1=path_to_reconstructed_depth / f'{name}.npy',
                path_to_depth_2=path_to_rendered_depth / f'query_{name}.npy',
                mm_to_m=True,
                name=f'{i+1}_depth_comparison',
                path_to_output=path_to_output,
            )
