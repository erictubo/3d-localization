import bpy
import os
import numpy as np
from math import pi, sqrt, atan2
from mathutils import Vector, Quaternion


class Blender:

    def __init__(self,
                 blend_file: str,
                 target_name: str,
                 camera_name: str,
                 output_dir: str,
                 target_size: list[float],
                 ):
        
        # Load CAD model
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

        self.target_size = target_size
        self.diagonal = sqrt(sum([s**2 for s in target_size]))

        self.prepare_depth_rendering()
        # self.prepare_edge_rendering()

    """
    CAMERA FUNCTIONS
    """
    
    def set_camera_intrinsics(self, focal_length, sensor_width, sensor_height):

        self.camera.data.lens = focal_length
        self.camera.data.sensor_width = sensor_width
        self.camera.data.sensor_height = sensor_height
    
    def get_camera_intrinsics(self):

        focal_length = self.camera.data.lens
        sensor_width = self.camera.data.sensor_width
        sensor_height = self.camera.data.sensor_height

        return focal_length, sensor_width, sensor_height

    def set_camera_pose(self, distance, h_angle, v_angle, height):

        print("Camera:", self.camera.location, self.camera.rotation_quaternion)

        tx, ty, tz = self.target.location + Vector((0, 0, height))

        print("Target:", tx, ty, tz)
    
        h_axis = Vector((0, 0, 1))
        h_quat = Quaternion(h_axis, h_angle)

        v_axis = Vector((0, -1, 0))
        v_quat = Quaternion(v_axis, v_angle)

        combined_quat = h_quat @ v_quat

        offset = Vector((distance, 0, 0))
        rotated_offset = combined_quat @ offset
        self.camera.location = Vector((tx, ty, tz)) + rotated_offset

        direction = Vector((tx, ty, tz)) - self.camera.location
        direction.normalize()
        quat_to_target = direction.to_track_quat('-Z', 'Y')
        self.camera.rotation_quaternion = quat_to_target
    
    def get_camera_pose(self):
        "get camera pose as distance, h_angle, v_angle, height with respect to target"
        pass

    def save_camera_pose(self, id:str, format='quaternion'):
        "save camera pose as text file with px, py, pz, qx, qy, qz, qw"

        if format == 'quaternion':
            px, py, pz = self.camera.location
            qx, qy, qz, qw = self.camera.rotation_quaternion
            pose = [px, py, pz, qx, qy, qz, qw]
            pose = ' '.join(map(str, pose))
            with open(os.path.join(self.output_dir, f'pose_{id}.txt'), 'w') as f:
                f.write(pose)

    """
    RENDERING FUNCTIONS
    """

    def prepare_depth_rendering(self):
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
    
    def render(self, id:str):

        image_file = os.path.join(self.output_dir, 'image_' + id)
        bpy.context.scene.render.filepath = image_file

        # Depth
        self.depth_output.file_slots[0].path = 'depth_' + id

        tx, ty, tz = self.target.location
        cx, cy, cz = self.camera.location
        distance = ((tx - cx)**2 + (ty - cy)**2 + (tz - cz)**2)**0.5
        min_depth = round(distance - self.diagonal/2 -1)
        max_depth = round(distance + self.diagonal/2 +1)
        self.map_range_node.inputs['From Min'].default_value = min_depth
        self.map_range_node.inputs['From Max'].default_value = max_depth
        
        # self.edges_output.file_slots[0].path = 'edges_' + id
        
        bpy.ops.render.render(write_still=True)
        self.save_camera_pose(id)
        self.save_depth_values(id)

        print(f"Distance: {round(distance)}, Min Depth: {min_depth}, Max Depth: {max_depth}")

        print("Camera:", self.camera.location, self.camera.rotation_quaternion)

    def render_orbit_view(self, distance: int, h_angle_deg: int, v_angle_deg: int, height: int, id: str):
        h_angle = h_angle_deg * pi/180
        v_angle = v_angle_deg * pi/180
        self.set_camera_pose(distance, h_angle, v_angle, height)
        self.render(id)

    def render_orbit_views(self, distances: list[int], h_steps: int, v_angles_deg: list[int], height: int):

        h_angles_deg = [deg for deg in np.linspace(0, 360, h_steps, endpoint=False)]
        # v_angles_deg = [v_angle for v_angle in np.linspace(v_min_deg, v_max_deg, v_steps)]

        i:int = 0

        for distance in distances:
            for v_angle_deg in v_angles_deg:
                for h_angle_deg in h_angles_deg:

                    i += 1
                    id = f'{i:04d}'
                    self.render_orbit_view(distance, h_angle_deg, v_angle_deg, height, id)
        
        self.save_combined_depth_values([f'{j:04d}' for j in range(1, i+1)])

    """
    DEPTH FUNCTIONS
    """
    
    def save_depth_values(self, id:str):
        "Calculate actual depth values from depth image and store in numpy array"

        white = self.map_range_node.inputs['From Min'].default_value
        black = self.map_range_node.inputs['From Max'].default_value

        depth_image = bpy.data.images.load(os.path.join(self.output_dir, f'depth_{id}0027.png'))
        depth_values = np.array(depth_image.pixels[:]).reshape(depth_image.size[1], depth_image.size[0], 4)

        depth_values = depth_values[:, :, 0]
        depth_values[depth_values == 0] = np.inf
        depth_values[depth_values != np.inf] = depth_values[depth_values != np.inf] * (white - black) + black

        assert depth_values.shape == (depth_image.size[1], depth_image.size[0])
        assert depth_values.min() > white
        assert depth_values[depth_values != np.inf].max() < black

        np.save(os.path.join(self.output_dir, f'depth_{id}.npy'), depth_values)

    def save_combined_depth_values(self, ids: list[str], name: str = 'depth'):
        "Combine depth value .npy files into a .npz file"

        depth_values_dict = {}
        for id in ids:
            depth_values = np.load(os.path.join(self.output_dir, f'depth_{id}.npy'))
            depth_values_dict[id] = depth_values

        print(f"Saving combined depth values to {name}.npz")
        np.savez(os.path.join(self.output_dir, name + '.npz'), **depth_values_dict)

    def get_depth_values_at_pixel(self, id:str, v:int, u:int, source='npy'):
        "Get depth values at a specific pixel location"

        if source == 'npy':
            depth_values = np.load(os.path.join(self.output_dir, f'depth_{id}.npy'))

        elif source == 'npz':
            depth_values = np.load(os.path.join(self.output_dir, 'depth.npz'))[id]

        return depth_values[v, u]


if __name__ == "__main__":

    model = 'notre dame'

    blender_dir = '/Users/eric/Library/Mobile Documents/com~apple~CloudDocs/Blender'

    output_dir = f'{blender_dir}/renders/{model}/'

    if model == 'notre dame':
        blend_file = f'{blender_dir}/assets/models/notre dame/notre dame.blend'
        target_name = 'SketchUp'
        target_size = [149, 114, 94.2]
        distance = 250
        height = target_size[2]/2

    if model == 'house luxury':    
        blend_file = f'{blender_dir}/assets/models/house luxury/house luxury.blend'
        target_name = 'House Luxury'
        target_size = [11.2, 11.3, 8]
        distance = 50
        height = target_size[2]/2

    if model == 'cube':
        blend_file = f'{blender_dir}/assets/models/cube.blend'
        target_name = 'Cube'
        target_size = [2, 2, 2]
        distance = 10
        height = 0

    blender = Blender(blend_file, target_name=target_name, camera_name='Camera', output_dir=output_dir, target_size=target_size)

    blender.render(id='0000')

    blender.render_orbit_views(distances=[distance], h_steps=24, v_angles_deg=[0], height=height)


# TODO
# - set camera instrinsics of blender camera to match real camera
# - automatically set depth range based on actual depth range
# - set bounding box of target object to automatically get target size (see code below)
# - position target to ground & rotate to align with axes
# - automatically find output number assigned by render
# - set start position (offset angle) for orbit view
# - re-position camera relatively

# When rendering images, Blender assigns a number to the image file name
# How can I get this number to save the camera pose and depth values with the same number?