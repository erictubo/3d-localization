import bpy
import os
import numpy as np
from math import pi, sqrt, atan2
from mathutils import Vector, Quaternion
import typing

from typing import Dict, Tuple, List, Optional, Union
from pathlib import Path



class RenderBlender:

    def __init__(self,
                 blend_file: str,
                 target_name: str,
                 camera_name: str,
                 output_dir: str,
                #  target_size: 'list[float]',
                 resolution: 'tuple[int]' = (1024, 1024),
                 depth_rendering: bool = True,
                 edge_rendering: bool = False
                 ):
        
        # TODO: Directly load CAD model
        # bpy.ops.import_scene.obj(filepath='path_to_your_model.obj')

        # Open existing blend file
        bpy.ops.wm.open_mainfile(filepath=blend_file)

        self.target = bpy.data.objects[target_name]
        self.camera = bpy.data.objects[camera_name]

        self.target.rotation_mode = 'QUATERNION'
        self.camera.rotation_mode = 'QUATERNION'

        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.center = self.find_target_center()
        self.target_size, self.diagonal = self.find_target_size()

        print(self.target_size)

        bpy.context.scene.render.resolution_x = resolution[0]
        bpy.context.scene.render.resolution_y = resolution[1]
        
        self.depth_rendering = depth_rendering
        if self.depth_rendering:
            self.prepare_depth_rendering()

        self.edge_rendering = edge_rendering
        if self.edge_rendering:
            self.prepare_edge_rendering()

    """
    CAMERA FUNCTIONS
    - Intrinsics
        - Get
        - Set
    - Pose: absolute, relative, orbit
        - Get
        - Set
        - Save
        - Retrieve
    """

    def get_camera_intrinsics(self) -> 'tuple[float, int, int]':
        """
        Get camera intrinsics (focal_length, sensor_width, sensor_height).
        """
        self.camera.data.lens_unit = 'FOV'
        fov_deg = self.camera.data.angle * 180/pi
        sensor_width = self.camera.data.sensor_width
        sensor_height = self.camera.data.sensor_height

        return fov_deg, sensor_width, sensor_height
    
    def set_camera_intrinsics(self, fov_deg: float, sensor_width: int, sensor_height: int):
        """
        Set camera intrinsics (focal_length, sensor_width, sensor_height).
        """
        self.camera.data.lens_unit = 'FOV'
        self.camera.data.angle = fov_deg * pi/180
        self.camera.data.sensor_width = sensor_width
        self.camera.data.sensor_height = sensor_height

        bpy.context.scene.render.resolution_x = sensor_width
        bpy.context.scene.render.resolution_y = sensor_height

    def get_camera_pose(self) -> np.ndarray:
        """
        Get camera pose (px, py, pz, qx, qy, qz, qw).
        """
        px, py, pz = self.camera.location
        qw, qx, qy, qz = self.camera.rotation_quaternion
        pose = np.array([px, py, pz, qw, qx, qy, qz])

        return pose

    def set_camera_pose(self, pose: np.ndarray):
        """
        Set camera pose (px, py, pz, qw, qx, qy, qz) in world coordinates.
        """
        px, py, pz, qw, qx, qy, qz = pose

        self.camera.location = Vector((px, py, pz))
        self.camera.rotation_quaternion = Quaternion((qw, qx, qy, qz))

    def save_camera_pose(self, id: str):
        """
        Save current camera pose as text file.
        """
        pose = self.get_camera_pose()

        with open(os.path.join(self.output_dir, f'pose_{id}.txt'), 'w') as f:
            f.write(' '.join(map(str, pose)))

    def retrieve_camera_pose(self, id:str) -> np.ndarray:
        """
        Retrieve camera pose from text file.
        """
        with open(os.path.join(self.output_dir, f'pose_{id}.txt'), 'r') as f:
            pose = f.read().split()
            pose = np.array([float(p) for p in pose])
        
        return pose
    
    def find_target_size(self) -> 'tuple[Vector, float]':
        """
        Find the bounding box target size in global frame.
        """
        obj = bpy.context.object
        global_bbox_corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]

        min_coords = [min(coords) for coords in zip(*global_bbox_corners)]
        max_coords = [max(coords) for coords in zip(*global_bbox_corners)]
    
        x_dim = max_coords[0] - min_coords[0]
        y_dim = max_coords[1] - min_coords[1]
        z_dim = max_coords[2] - min_coords[2]

        diagonal = sqrt(x_dim**2 + y_dim**2 + z_dim**2)
        
        return Vector([x_dim, y_dim, z_dim]), diagonal

    def find_target_center(self) -> Vector:
        """
        Find the center of geometry of the target object in global frame.
        """
        obj = bpy.context.object
        local_bbox_center = 0.125 * sum((Vector(corner) for corner in obj.bound_box), Vector())
        global_bbox_center = obj.matrix_world @ local_bbox_center

        return global_bbox_center

    def set_camera_orbit_pose(self, distance: float, h_angle: float, v_angle: float):
        """
        Set camera pose in orbit view around target object.
        Parameters:
        - distance: distance from target object
        - h_angle: horizontal angle in radians
        - v_angle: vertical angle in radians
        """
    
        h_axis = Vector((0, 0, 1))
        h_quat = Quaternion(h_axis, h_angle)

        v_axis = Vector((0, -1, 0))
        v_quat = Quaternion(v_axis, v_angle)

        combined_quat = h_quat @ v_quat

        offset = Vector((distance, 0, 0))
        rotated_offset = combined_quat @ offset
        self.camera.location = self.center + rotated_offset

        direction = self.center - self.camera.location
        direction.normalize()
        quat_to_target = direction.to_track_quat('-Z', 'Y')
        self.camera.rotation_quaternion = quat_to_target

    def rotate_camera(roll: float, pitch: float, yaw: float):
        """
        Rotate camera around itself.
        """
        # Angles are interdependent, so sequence matters.
        # Use quaternions instead.
        pass

    def rotate_camera_relative_to_target(h_angle: float, v_angle: float, roll: float):
        """
        Rotate camera relative to target object.
        """
        pass

    def reposition_camera_relative_to_target(distance: float, height: float, ):
        """
        Reposition camera relative to target object.
        """
        pass

    """
    DEPTH & EDGE SET-UP FUNCTIONS
    """

    def prepare_depth_rendering(self):
        """
        Prepare settings for depth rendering:
        - Enable depth pass
        - Create map range node to normalize depth
        - Create output nodes for depth image
        """
        view_layer = bpy.context.scene.view_layers[0]
        view_layer.use_pass_z = True

        bpy.context.scene.use_nodes = True
        tree = bpy.context.scene.node_tree
        links = tree.links

        # Clear existing nodes
        for node in tree.nodes:
            tree.nodes.remove(node)

        # Create render layers node
        self.render_layers = tree.nodes.new(type="CompositorNodeRLayers")

        # Create map range node to normalize depth
        self.map_range_node = tree.nodes.new(type="CompositorNodeMapRange")
        # self.map_range_node.inputs['From Min'].default_value = min_depth
        # self.map_range_node.inputs['From Max'].default_value = max_depth
        self.map_range_node.inputs['To Min'].default_value = 1.0
        self.map_range_node.inputs['To Max'].default_value = 0.0

        # Create output nodes for depth image
        self.depth_output = tree.nodes.new(type="CompositorNodeOutputFile")
        self.depth_output.label = "Depth Output"
        self.depth_output.base_path = self.output_dir
        self.depth_output.file_slots[0].path = "depth_####"
        self.depth_output.format.file_format = 'PNG'
        self.depth_output.format.color_mode = 'BW'

        # Link nodes
        links.new(self.render_layers.outputs['Depth'], self.map_range_node.inputs[0])
        links.new(self.map_range_node.outputs[0], self.depth_output.inputs[0])
    
    def prepare_edge_rendering(self):
        """
        Prepare settings for edge rendering:
        - Enable Freestyle rendering
        - Set line thickness
        - Create Freestyle line set
        - Create output nodes for edge image
        """
        scene = bpy.context.scene
        scene.render.use_freestyle = True
        scene.render.line_thickness_mode = 'ABSOLUTE'
        scene.render.line_thickness = 1.0
        
        # Create Freestyle line set
        view_layer = bpy.context.view_layer
        freestyle_settings = view_layer.freestyle_settings
        lineset = freestyle_settings.linesets.new("LineSet")
        freestyle_settings.as_render_pass = True
        lineset.select_silhouette = True
        lineset.select_border = True
        lineset.select_crease = True

        # Ensure that only the edges are rendered
        bpy.context.scene.render.film_transparent = True

        # Create output nodes for edge image
        tree = scene.node_tree
        links = tree.links

        # Clear existing nodes
        for node in tree.nodes:
            if node.label == "Edges Output":
                tree.nodes.remove(node)

        # Create new render layers node to include Freestyle output
        freestyle_render_layers = tree.nodes.new(type="CompositorNodeRLayers")
        freestyle_render_layers.name = "Freestyle Render Layers"

        # Create a compositor node to output the edges image
        self.edges_output = tree.nodes.new(type="CompositorNodeOutputFile")
        self.edges_output.label = "Edges Output"
        self.edges_output.base_path = self.output_dir
        self.edges_output.file_slots[0].path = 'edges_####'
        self.edges_output.format.file_format = 'PNG'
        self.edges_output.format.color_mode = 'RGBA'

        # Link the Freestyle edge pass to the edges output node
        links.new(freestyle_render_layers.outputs['Freestyle'], self.edges_output.inputs[0])
    
    """
    DEPTH FUNCTIONS
    """
    
    def save_depth_values(self, id:str):
        """
        Calculate actual depth values from depth image and store in numpy array.
        """
        white = self.map_range_node.inputs['From Min'].default_value
        black = self.map_range_node.inputs['From Max'].default_value

        depth_image = bpy.data.images.load(os.path.join(self.output_dir, f'depth_{id}0027.png'))
        depth_values = np.array(depth_image.pixels[:]).reshape(depth_image.size[1], depth_image.size[0], 4)

        depth_values = depth_values[:, :, 0]
        depth_values[depth_values == 0] = np.inf
        depth_values[depth_values != np.inf] = depth_values[depth_values != np.inf] * (white - black) + black

        assert depth_values.shape == (depth_image.size[1], depth_image.size[0])
        assert depth_values.min() > white
        # assert depth_values[depth_values != np.inf].max() < black

        np.save(os.path.join(self.output_dir, f'depth_{id}.npy'), depth_values)

    def save_combined_depth_values(self, ids: List[str], name: str = 'depth'):
        """
        Combine depth values NPY files per image into a single NPZ file for all images.
        """
        depth_values_dict = {}
        for id in ids:
            depth_values = np.load(os.path.join(self.output_dir, f'depth_{id}.npy'))
            depth_values_dict[id] = depth_values

        print(f"Saving combined depth values to {name}.npz")
        np.savez(os.path.join(self.output_dir, name + '.npz'), **depth_values_dict)

    def get_depth_values_at_pixel(self, id:str, v:int, u:int, source='npy') -> float:
        """
        Get depth value at a specific pixel location.
        """
        if source == 'npy':
            depth_values = np.load(os.path.join(self.output_dir, f'depth_{id}.npy'))

        elif source == 'npz':
            depth_values = np.load(os.path.join(self.output_dir, 'depth.npz'))[id]

        return depth_values[v, u]
    
    """
    RENDERING FUNCTIONS
    """
    
    def render(self, id:str):
        """
        Render images (+ depth and edges if enabled), save camera pose and depth values.
        """
        image_file = os.path.join(self.output_dir, f'image_{id}.png')
        bpy.context.scene.render.filepath = image_file

        # Edge rendering
        if self.edge_rendering:
            self.edges_output.file_slots[0].path = 'edges_' + id
        
        # Depth rendering
        if self.depth_rendering:
            self.depth_output.file_slots[0].path = 'depth_' + id

        bpy.ops.render.render(write_still=True)

        if self.depth_rendering:
            # Set depth range based on distance to target
            distance = sqrt(sum((self.center-self.camera.location) * (self.center-self.camera.location)))
            min_depth = round(distance - self.diagonal/2 -1)
            max_depth = round(distance + self.diagonal/2 +1)
            self.map_range_node.inputs['From Min'].default_value = min_depth
            self.map_range_node.inputs['From Max'].default_value = max_depth
            
            # Save depth values
            self.save_depth_values(id)

        # Save camera pose
        self.save_camera_pose(id)

    def render_orbit_view(self, distance: int, h_angle_deg: int, v_angle_deg: int, id: str):
        """
        Render orbit view at a specific distance, horizontal and vertical angle (deg).
        """
        h_angle = h_angle_deg * pi/180
        v_angle = v_angle_deg * pi/180
        self.set_camera_orbit_pose(distance, h_angle, v_angle)
        self.render(id)

    def render_orbit_views(self, distances: 'list[int]', h_steps: int, v_angles_deg: 'list[int]'):
        """
        Render orbit views at multiple distances, horizontal and vertical angles (deg).
        """
        h_angles_deg = [deg for deg in np.linspace(0, 360, h_steps, endpoint=False)]

        assert min(distances) > max(self.target_size)/2, \
            f'minimum distance {min(distances)} less than half of largest target size {max(self.target_size)}'

        i:int = 0
        for distance in distances:
            for v_angle_deg in v_angles_deg:
                for h_angle_deg in h_angles_deg:

                    i += 1
                    id = f'{i:04d}'
                    self.render_orbit_view(distance, h_angle_deg, v_angle_deg, id)

        print(f"Rendered {i} images")
        
        self.save_combined_depth_values([f'{j:04d}' for j in range(1, i+1)])

    def retrieve_query_data(path_to_outputs: Path, query_name: str) -> Tuple[Tuple[str, np.ndarray, int, int], np.ndarray]:
        """
        Retrieve query data (intrinsics, pose) from file.
        """
        with open(path_to_outputs + query_name.replace('.jpg', '.txt').replace('query/',''), 'r') as f:
            data = f.readlines()
            camera_model = data[0].strip()
            params = np.array(data[1].strip().split(), dtype=float)
            w, h = map(int, data[2].strip().split())
            pose = np.array(data[3].strip().split(), dtype=float)
            assert pose.shape == (7,), pose.shape

        return (camera_model, params, w, h), pose

    def get_field_of_view_from_intrinsics(camera_model: str, camera_params: np.ndarray, w: int, h: int) -> float:
        """
        Compute field of view (FoV) in degrees.
        """
        if camera_model == 'PINHOLE':
            assert camera_params.shape == (4,), camera_params.shape
            fx, fy, cx, cy = camera_params

            if w > h:   # FoV along horizontal axis
                fov = 2 * np.arctan(w / (2 * fx))
                fov_deg = fov * 180/pi

            else:       # FoV along vertical axis
                fov = 2 * np.arctan(h / (2 * fy))
                fov_deg = fov * 180/pi
        else:
            raise ValueError(f"Camera model {camera_model} not implemented.")

        return fov_deg
    
    def render_query_poses(self, path_to_images: Path, path_to_outputs: Path):
        """
        Render query images with pose and intrinsics.
        """
        query_names = [f.name for f in path_to_images.glob('query/*.jpg')]

        for query_name in query_names:
            (camera_model, camera_params, w, h), pose = self.retrieve_query_data(path_to_outputs / 'query_data/', query_name)
            print('Camera model:', camera_model)
            print('Camera params:', camera_params)
            print('Image size:', w, h)
            print('Pose:', pose)

            fov = self.get_field_of_view_from_intrinsics(camera_model, camera_params, w, h)

            print(f"Query: {query_name}, FoV: {fov:.2f} deg")

            blender.set_camera_pose(pose)
            blender.set_camera_intrinsics(fov, w, h)

            id = f'test_{query_name.replace(".jpg", "")}'
            blender.render(id=id)


if __name__ == "__main__":


    blender_dir = '/Users/eric/Library/Mobile Documents/com~apple~CloudDocs/Blender/'

    evaluation_dir = '/Users/eric/Downloads/evaluation/'
    
    models: dict = {
        # 'cube':{
        #     'blend_file': f'{blender_dir}assets/models/cube.blend',
        #     'target_name': 'Cube',
        #     'target_size': [2, 2, 2],
        #     'distances': [10],
        # },
        # 'house luxury':{
        #     'blend_file': f'{blender_dir}assets/models/house luxury/house luxury.blend',
        #     'target_name': 'House Luxury',
        #     'target_size': [11.2, 11.3, 8],
        #     'distances': [50],
        # },
        # 'notre dame B':{
        #     'blend_file': f'{blender_dir}assets/models/notre dame B/notre dame B.blend',
        #     'target_name': 'SketchUp',
        #     'target_size': [149, 114, 94.2],
        #     'distances': [100, 150, 200],
        # },
        # 'notre dame C':{
        #     'blend_file': f'{blender_dir}/assets/models/notre dame/notre dame C.blend',
        #     'target_name': 'SketchUp',
        #     'target_size': [149, 114, 94.2],
        # },
        'notre dame E':{
            'blend_file': f'{blender_dir}assets/models/notre dame E/notre dame E.blend',
            'target_name': 'notre-dame-de-paris-complete-miniworld3d',
            # 'target_size': [71.5, 165, 105],
            'distances': [100, 150, 200],
        }
    }

    h_steps = 24 # 260/24 = 15 [deg]
    v_angles_deg = [-15, 0, 15] # [deg]

    for model in models:

        output_dir = f'{blender_dir}renders/{model}/'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        blend_file = models[model]['blend_file']
        target_name = models[model]['target_name']
        # target_size = models[model]['target_size']
        distances = models[model]['distances']
        
        # height = target_size[2]/2


        blender = RenderBlender(
            blend_file,
            target_name=target_name,
            camera_name='Camera',
            output_dir=output_dir,
            # target_size=target_size,
            )
        


        # blender.render(id='0000')

        blender.render_orbit_views(
            distances=distances,
            h_steps=h_steps,
            v_angles_deg=v_angles_deg,
            )


        # path_to_dataset = path_to_evaluation / model / '/'
        # path_to_images = path_to_dataset / 'images/'
        # path_to_outputs = path_to_dataset / 'outputs/'
        
        # blender.render_query_poses(
        #     path_to_images=path_to_images,
        #     path_to_outputs=path_to_outputs,
        #     )





# TODO: why are some images so different?
    # - intrinsics?
    # - pose?
    # - camera model?
    # - inaccuracy in data?

# TODO: run Blender code without task configuration
    

        

