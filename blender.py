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
                 ):
        
        # Load CAD model
        # bpy.ops.import_scene.obj(filepath='path_to_your_model.obj')

        # Open existing blend file
        bpy.ops.wm.open_mainfile(filepath=blend_file)

        self.target = bpy.data.objects[target_name]
        self.camera = bpy.data.objects[camera_name]

        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.prepare_depth_rendering()
        self.prepare_edge_rendering()


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
        self.map_range_node.inputs['From Min'].default_value = 9.5
        self.map_range_node.inputs['From Max'].default_value = 12.5
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
        self.edges_output.file_slots[0].path = "edges_####"
        self.edges_output.format.file_format = 'PNG'
        self.edges_output.format.color_mode = 'RGBA'

        # Link the Freestyle edge pass to the edges output node
        links.new(freestyle_render_layers.outputs['Freestyle'], self.edges_output.inputs[0])

    
    def render(self, image_name:str='image', depth=True, depth_name:str='depth', edges=True, edges_name:str='edges'):

        image_file = os.path.join(self.output_dir, image_name)
        bpy.context.scene.render.filepath = image_file

        if depth:
            self.depth_output.file_slots[0].path = depth_name

        if edges:
            self.edges_output.file_slots[0].path = edges_name
        
        bpy.ops.render.render(write_still=True)


    def set_camera_intrinsics(self, focal_length, sensor_width, sensor_height):

        self.camera.data.lens = focal_length
        self.camera.data.sensor_width = sensor_width
        self.camera.data.sensor_height = sensor_height


    def set_camera_pose(self, distance, h_angle, v_angle, height):

        tx, ty, tz = self.target.location + Vector((0, 0, height))
    
        h_axis = Vector((0, 0, 1))
        h_quat = Quaternion(h_axis, h_angle)

        v_axis = Vector((0, 1, 0))
        v_quat = Quaternion(v_axis, v_angle)

        combined_quat = h_quat @ v_quat

        offset = Vector((distance, 0, 0))
        rotated_offset = combined_quat @ offset
        self.camera.location = Vector((tx, ty, tz)) - rotated_offset

        direction = Vector((tx, ty, tz)) - self.camera.location
        direction.normalize()
        quat_to_target = direction.to_track_quat('-Z', 'Y')
        self.camera.rotation_quaternion = quat_to_target


    def render_from_different_views(self,
                                    distances: list[int],
                                    h_steps: int,
                                    v_angles_deg: list[int],
                                    height: int,
                                    ):

        h_angles_deg = [deg for deg in np.linspace(0, 360, h_steps, endpoint=False)]
        # v_angles_deg = [v_angle for v_angle in np.linspace(v_min_deg, v_max_deg, v_steps)]

        self.prepare_depth_rendering()

        for d in distances:
            for v in v_angles_deg:
                v_angle = v * pi/180
                for h in h_angles_deg:
                    h_angle = h * pi/180

                    self.set_camera_pose(d, h_angle, v_angle, height)
                    self.render(f'image_{d}_{v}_{h}')

                    # ex, ey, ez = self.camera.rotation_euler
                    # print(self.camera.location, ex * 180/pi, ey * 180/pi, ez * 180/pi)


if __name__ == "__main__":
    blend_file = '/Users/eric/Blender/assets/models/test.blend'
    output_directory = '/Users/eric/Downloads/Blender/depth_images/'
    blender = Blender(blend_file, target_name='Cube', camera_name='Camera', output_dir=output_directory)
    # blender.render_from_different_views()
    blender.render()


# TODO
# - set camera instrinsics of blender camera to match real camera
# - calculate actual depth values from depth image
# - automatically set depth range based on actual depth range