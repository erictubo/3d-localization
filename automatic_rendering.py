import bpy
import os
import csv
import numpy as np
from math import pi, sqrt, atan2
from mathutils import Quaternion, Vector
import bpy_extras.view3d_utils as view3d_utils


class AutomaticRendering:

    def __init__(self, blend_file, target_name, camera_name, output_dir, distances=[20, 40], h_steps=4, v_steps=1, v_max_deg=60, z_offset=1.7):

        # Set up data
        bpy.ops.wm.open_mainfile(filepath=blend_file)
        self.target = bpy.data.objects[target_name]
        self.camera = bpy.data.objects[camera_name]
        self.output_dir = output_dir

        self.setup_output_directory()

        # Set up rendering parameters
        self.distances = distances
        self.horizontal_angles_deg = [horizontal_angle for horizontal_angle in range(0, 360, 360//h_steps)]
        self.vertical_angles_deg = [vertical_angle for vertical_angle in range(0, v_max_deg, v_max_deg//v_steps)]
        self.z_offset = z_offset
        
        # Prepare depth rendering
        self.enable_depth_pass()
        self.setup_scene_nodes()


    def setup_output_directory(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def enable_depth_pass(self):
        view_layer = bpy.context.scene.view_layers[0]
        view_layer.use_pass_z = True

    def setup_scene_nodes(self):
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
        # self.map_range_node.inputs['From Min'].default_value = 1.0
        # self.map_range_node.inputs['From Max'].default_value = 0.0

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

    def calculate_distances_and_angles(self):
        tx, ty, tz = self.target.location.x, self.target.location.y, self.target.location.z + self.z_offset
        cx, cy, cz = self.camera.location.x, self.camera.location.y, self.camera.location.z

        init_distance = sqrt((tx-cx)**2 + (ty-cy)**2 + (tz-cz)**2)
        init_distance_xy = sqrt((tx-cx)**2 + (ty-cy)**2)
        init_h_rot_angle = atan2(cy-ty, cx-tx)
        init_v_rot_angle = atan2(cz-tz, init_distance_xy)

        init_h_rot_angle = 7.5 * pi/180
        init_v_rot_angle = 0 * pi/180

        return tx, ty, tz, init_distance, init_h_rot_angle, init_v_rot_angle


    def set_camera_position_and_rotation(self, tx, ty, tz, distance, v_rot_angle, h_rot_angle):
        v_rot_axis = Vector((0, 1, 0))
        v_rot_quat = Quaternion(v_rot_axis, v_rot_angle)

        h_rot_axis = Vector((0, 0, 1))
        h_rot_quat = Quaternion(h_rot_axis, h_rot_angle)

        combined_rot_quat = h_rot_quat @ v_rot_quat

        offset = Vector((distance, 0, 0))
        rotated_offset = combined_rot_quat @ offset
        self.camera.location = Vector((tx, ty, tz)) - rotated_offset

        direction = Vector((tx, ty, tz)) - self.camera.location
        direction.normalize()
        quat_to_target = direction.to_track_quat('-Z', 'Y')
        self.camera.rotation_quaternion = quat_to_target

    def render_images(self):
        tx, ty, tz, init_distance, init_h_rot_angle, init_v_rot_angle = self.calculate_distances_and_angles()

        for d in self.distances:
            for v in self.vertical_angles_deg:
                vertical_angle = v * pi/180
                for h in self.horizontal_angles_deg:
                    horizontal_angle = h * pi/180

                    self.set_camera_position_and_rotation(tx, ty, tz, d, vertical_angle, horizontal_angle)

                    image_file = os.path.join(self.output_dir, f'image_{d}_{v}_{h}')
                    # depth_file = os.path.join(self.output_dir, f'depth_{d}_{v}_{h}.npy')
                    self.depth_output.file_slots[0].path = f'depth_{d}_{v}_{h}'
                    bpy.context.scene.render.filepath = image_file

                    bpy.ops.render.render(write_still=True)
                    
                    # DepthOperations.save_depth_to_numpy(self.camera, depth_file)
                    # DepthOperations.save_depth_to_csv(depth_file, np.load(depth_file))

                    # # Normalize depth map
                    # DepthOperations.normalize_depth_map(depth_file, self.output_dir)


class DepthOperations:
    @staticmethod
    def save_depth_to_numpy(camera, depth_file):
        scene = bpy.context.scene

        # Get the depth pass
        result = bpy.data.images['Render Result']
        width = result.size[0]
        height = result.size[1]

        # Convert Z-pass values to actual depth values in meters
        coords = np.indices((width, height)).reshape(2, -1).T
        depth_in_meters = np.zeros((height, width))

        for coord in coords:
            x, y = coord
            view_vector = view3d_utils.region_2d_to_vector_3d(scene.render.layers.active, (x, y))
            world_coords = view3d_utils.region_2d_to_location_3d(scene.render.layers.active, camera.matrix_world.to_quaternion(), view_vector)
            depth_in_meters[y, x] = (world_coords - camera.location).length

        # Save depth data as a numpy array
        np.save(depth_file, depth_in_meters)

    @staticmethod
    def save_depth_to_csv(depth_file, depth_data):
        csv_file = depth_file.replace('.npy', '.csv')
        np.savetxt(csv_file, depth_data, delimiter=',')

    @staticmethod
    def normalize_depth_map(depth_file, output_dir):
        # Load depth data
        depth_data = np.load(depth_file)

        # Handle infinity values
        finite_depth_data = depth_data[np.isfinite(depth_data)]

        # Check if the array is not empty
        if finite_depth_data.size == 0:
            print("Warning: No valid depth data found. Skipping normalization for this frame.")
            return

        # Find min and max depth values ignoring infinities
        min_depth = np.min(finite_depth_data)
        max_depth = np.max(finite_depth_data)

        # Normalize the depth data
        normalized_depth = (depth_data - min_depth) / (max_depth - min_depth)

        # Get the height and width from the depth data shape
        height, width = depth_data.shape

        # Prepare a new image to store normalized depth values
        normalized_image = np.zeros((height, width, 4), dtype=np.float32)
        normalized_image[:, :, 0] = normalized_depth
        normalized_image = normalized_image.flatten()

        # Create a new image in Blender to store the normalized depth map
        normalized_depth_image = bpy.data.images.new("Normalized Depth", width=width, height=height, alpha=True)
        normalized_depth_image.pixels = normalized_image

        # Save the normalized depth image
        normalized_depth_image.filepath_raw = os.path.join(output_dir, os.path.basename(depth_file).replace('.npy', '_normalized.png'))
        normalized_depth_image.file_format = 'PNG'
        normalized_depth_image.save()


if __name__ == "__main__":
    blend_file = '/Users/eric/Blender/assets/models/house luxury/house luxury.blend'
    output_directory = '/Users/eric/Downloads/Blender/depth_images/'
    distance_steps = [20, 40]
    blender_render = AutomaticRendering(blend_file, target_name='House Luxury', camera_name='Camera', output_dir=output_directory)
    blender_render.render_images()

# IDEAS
# - first calculate depth values, then render depth image using normalized depth values


# TODO
# - set camera instrinsics of blender camera to match real camera





## Set up rendering of depth map:
#bpy.context.scene.use_nodes = True
#tree = bpy.context.scene.node_tree
#links = tree.links

## clear default nodes
#for n in tree.nodes:
#    tree.nodes.remove(n)

## create input render layer node
#rl = tree.nodes.new('CompositorNodeRLayers')

#map = tree.nodes.new(type="CompositorNodeMapValue")
## Size is chosen kind of arbitrarily, try out until you're satisfied with resulting depth map.
#map.size = [0.08]
#map.use_min = True
#map.min = [0]
#map.use_max = True
#map.max = [255]
#links.new(rl.outputs[2], map.inputs[0])

#invert = tree.nodes.new(type="CompositorNodeInvert")
#links.new(map.outputs[0], invert.inputs[1])

## The viewer can come in handy for inspecting the results in the GUI
#depthViewer = tree.nodes.new(type="CompositorNodeViewer")
#links.new(invert.outputs[0], depthViewer.inputs[0])
## Use alpha from input.
#links.new(rl.outputs[1], depthViewer.inputs[1])


## create a file output node and set the path
#fileOutput = tree.nodes.new(type="CompositorNodeOutputFile")
#fileOutput.base_path = "/Users/eric/Downloads/Blender/depth/"
#links.new(invert.outputs[0], fileOutput.inputs[0])




# scn = bpy.context.scene
# cam = scn.camera
# output_path = scn.render.filepath
# tree = bpy.context.scene.node_tree
# links = tree.links

# bpy.context.scene.render.use_compositing = True
# bpy.context.scene.use_nodes = True
# bpy.context.scene.view_layers[0].use_pass_z = True

# for n in tree.nodes:
#   tree.nodes.remove(n)
# rl = tree.nodes.new('CompositorNodeRLayers')

# vl = tree.nodes.new('CompositorNodeViewer')
# vl.use_alpha = True
# links.new(rl.outputs[0], vl.inputs[0])  # link Image to Viewer Image RGB
# links.new(rl.outputs['Depth'], vl.inputs[1])  # link Render Z to Viewer Image Alpha


# bpy.context.scene.render.filepath = "/Users/eric/Downloads/Blender/depth/image"
# bpy.ops.render.render(write_still=True)


# pixels = np.array(bpy.data.images['Viewer Node'].pixels)
# resolution = int(np.sqrt(len(pixels)/4))
# #reshaping into image array 4 channel (rgbz)
# image_with_depth = pixels.reshape(resolution,resolution,4)