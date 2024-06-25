import bpy
import os
import numpy as np
from math import pi, sqrt, asin, acos, atan2
from mathutils import Vector, Quaternion
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Union


class Blender:

    def __init__(self,
                 blend_file: str,
                 render_dir: str,
                 target_name: str,
                 camera_name: str = 'Camera',
                 image_size: 'tuple[int]' = (1024, 1024),
                 depth_rendering: bool = True,
                 edge_rendering: bool = False,
                 images_prefix: str = 'images/',
                 depth_prefix: str = 'depth/',
                 poses_prefix: str = 'poses/',
                #  edges_prefix: str = 'edges/'
                 ):

        # Open blend file
        bpy.ops.wm.open_mainfile(filepath=blend_file)

        self.render_dir = render_dir

        self.images_prefix = images_prefix
        self.depth_prefix = depth_prefix
        self.poses_prefix = poses_prefix
        # self.edges_prefix = edges_prefix
        self.images_dir = render_dir + images_prefix
        self.depth_dir = render_dir + depth_prefix
        self.poses_dir = render_dir + poses_prefix
        # self.edges_dir = render_dir + edges_prefix

        for dir in [self.render_dir, self.images_dir, self.depth_dir, self.poses_dir]:
            if not os.path.exists(dir): os.makedirs(dir)
        
        self.target = bpy.data.objects[target_name]
        self.camera = bpy.data.objects[camera_name]

        self.target.rotation_mode = 'QUATERNION'
        self.camera.rotation_mode = 'QUATERNION'

        self.target_center, self.target_size, self.target_diagonal = \
            self.find_target_center_and_size()

        print(f"Target center: {self.target_center}")
        print(f"Target size: {self.target_size}")
        print(f"Diagonal: {self.target_diagonal}")

        self.distances = self.calculate_distances()

        print("Distances:", self.distances)

        w, h = image_size[0], image_size[1]
        _, _, f = self.get_camera_intrinsics('mm')
        self.set_camera_intrinsics(w, h, f, 'mm')
        
        self.depth_rendering = depth_rendering
        if self.depth_rendering:
            self.prepare_depth_rendering()

        self.edge_rendering = edge_rendering
        if self.edge_rendering:
            self.prepare_edge_rendering()


    """
    GEOMETRIC
    - target center
    - target size
    """

    def find_target_center_and_size(self):
        """
        Find the center of geometry and size of the target object or Empty in global frame.
        """
        obj = self.target
        
        if obj.type == 'EMPTY':
            # For Empty objects, consider all children
            children = [child for child in obj.children_recursive if child.type == 'MESH']
            if not children:
                return Vector((0, 0, 0)), Vector((0, 0, 0)), 0
            
            all_corners = []
            for child in children:
                child_corners = [child.matrix_world @ Vector(corner) for corner in child.bound_box]
                all_corners.extend(child_corners)
        else:
            # For normal objects, use its own bounding box
            all_corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
        
        # Calculate center and size
        min_coords = [min(coords) for coords in zip(*all_corners)]
        max_coords = [max(coords) for coords in zip(*all_corners)]
        
        center = Vector([(min_coord + max_coord) / 2 for min_coord, max_coord in zip(min_coords, max_coords)])
        
        x_dim = max_coords[0] - min_coords[0]
        y_dim = max_coords[1] - min_coords[1]
        z_dim = max_coords[2] - min_coords[2]
        
        size = Vector([x_dim, y_dim, z_dim])
        diagonal = sqrt(x_dim**2 + y_dim**2 + z_dim**2)
        
        assert diagonal != 0, "Diagonal is zero"
        
        return center, size, diagonal
    
    def calculate_distances(self) -> List[int]:
        """
        Calculate distance values for rendering based on target size and diagonal.
        """
        d = np.ceil(self.target_diagonal/2/10)*10

        return [d, 2*d]


    """
    CAMERA
    - intrinsics
    - pose
    - orbit pose
    """

    def get_camera_intrinsics(self, unit: str = 'mm') -> 'tuple[float, int, int]':
        """
        Get camera intrinsics (focal_length, sensor_width, sensor_height).
        """
        w = bpy.context.scene.render.resolution_x
        h = bpy.context.scene.render.resolution_y

        if unit.upper() == 'FOV':
            fov_deg = self.camera.data.angle * 180/pi
            return w, h, fov_deg
        elif unit.upper() == 'MM':
            f = self.camera.data.lens
            return w, h, f
        else:
            raise ValueError(f"Select valid unit: 'mm' or 'fov'")

    def set_camera_intrinsics(self, w: int, h: int, value: float, unit: str):
        """
        Set camera intrinsics (focal_length, w, h).
        """
        bpy.context.scene.render.resolution_x = w
        bpy.context.scene.render.resolution_y = h

        if unit.upper() == 'MM':
            self.camera.data.lens = value
        elif unit.upper() == 'FOV':
            self.camera.data.angle = value * pi/180   
        else:
            raise ValueError(f"Select valid unit: 'mm' or 'fov'") 

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

    def adjust_camera_pose(self, relative_pose: np.ndarray):
        """
        Adjust camera pose relative to its current pose.
        """
        t = Vector(relative_pose[:3])
        q = Quaternion(relative_pose[3:])
        self.camera.location = self.camera.location + t
        self.camera.rotation_quaternion = q * self.camera.rotation_quaternion

    def write_camera_pose(self, id: str):
        """
        Write current camera pose to a text file.
        """
        pose = self.get_camera_pose()


        with open(os.path.join(self.poses_dir, f'{id}.txt'), 'w') as f:
            f.write(' '.join(map(str, pose)))

    def read_camera_pose(self, id:str) -> np.ndarray:
        """
        Read camera pose from text file.
        """
        with open(os.path.join(self.poses_dir, f'{id}.txt'), 'r') as f:
            pose = f.read().split()
            pose = np.array([float(p) for p in pose])
        
        return pose

    def set_camera_orbit_pose(self, distance: float, h_angle: float, v_angle: float, unit: str = 'rad'):
        """
        Set camera pose in orbit view around target object.
        """
        if unit == 'deg':
            h_angle *= pi/180
            v_angle *= pi/180
    
        h_axis = Vector((0, 0, 1))
        h_quat = Quaternion(h_axis, h_angle)

        v_axis = Vector((0, -1, 0))
        v_quat = Quaternion(v_axis, v_angle)

        combined_quat = h_quat @ v_quat

        offset = Vector((distance, 0, 0))
        rotated_offset = combined_quat @ offset
        self.camera.location = self.target_center + rotated_offset

        direction = self.target_center - self.camera.location
        direction.normalize()
        quat_to_target = direction.to_track_quat('-Z', 'Y')
        self.camera.rotation_quaternion = quat_to_target

    def get_camera_orbit_pose(self) -> Tuple[float, float, float]:
        """
        Get camera pose in orbit view around object.
        """
        t = self.camera.location - self.target_center
        distance = sqrt(sum(t * t))

        x, y, z = t[0], t[1], t[2]

        h_angle = atan2(y, x)
        v_angle = asin(z/distance)
        
        return distance, h_angle, v_angle

    def adjust_camera_orbit_pose(
            self,
            relative_distance: float,
            relative_h_angle: float,
            relative_v_angle: float,
            unit: str = 'rad'):
        """
        Adjust camera pose in orbit view around target object.
        """
        if unit == 'deg':
            h_angle *= pi/180
            v_angle *= pi/180
        
        distance, h_angle, v_angle = self.get_camera_orbit_pose()

        distance += relative_distance
        h_angle += relative_h_angle
        v_angle += relative_v_angle

        self.set_camera_orbit_pose(distance, h_angle, v_angle)


    """
    DEPTH & EDGE RENDERING SET-UP
    """

    def prepare_depth_rendering(self):
        """
        Prepare settings for depth rendering.
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

        # Create output nodes for depth image
        self.depth_output = tree.nodes.new(type="CompositorNodeOutputFile")
        self.depth_output.base_path = self.depth_dir
        self.depth_output.file_slots[0].path = "####"
        self.depth_output.format.file_format = 'OPEN_EXR'

        # Link nodes
        links.new(self.render_layers.outputs['Depth'], self.depth_output.inputs[0])

    def prepare_edge_rendering(self):
        """
        Prepare settings for edge rendering.
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
        self.edges_output.base_path = self.edges_dir
        self.edges_output.file_slots[0].path = '####'
        self.edges_output.format.file_format = 'PNG'
        self.edges_output.format.color_mode = 'RGBA'

        # Link the Freestyle edge pass to the edges output node
        links.new(freestyle_render_layers.outputs['Freestyle'], self.edges_output.inputs[0])


    """
    RENDERING
    """
    
    def render(self, id:str):
        """
        Render images (+ depth and edges if enabled), save camera pose and depth values.
        """
        image_file = os.path.join(self.images_dir, f'{id}.png')
        bpy.context.scene.render.filepath = image_file

        if self.edge_rendering:
            self.edges_output.file_slots[0].path = id
        
        self.depth_output.file_slots[0].path = id

        bpy.ops.render.render(write_still=True)

        self.write_camera_pose(id)

    def render_orbit_view(self, h_angle_deg: int, v_angle_deg: int, distance: int, id: str):
        """
        Render orbit view at a specific distance, horizontal and vertical angle (deg).
        """
        assert distance > self.target_diagonal/2, \
            f' distance {distance} less than half of target diagonal {self.target_diagonal/2}'
        assert abs(v_angle_deg) <= 75, \
            f'vertical angle {v_angle} greater than +/-75 [deg]'

        h_angle = h_angle_deg * pi/180
        v_angle = v_angle_deg * pi/180
        self.set_camera_orbit_pose(distance, h_angle, v_angle)
        self.render(id)

    def render_orbit_views(self, h_steps: int, v_angles_deg: 'list[int]', distances: 'list[int]' = None):
        """
        Render orbit views at multiple distances, horizontal and vertical angles (deg).
        """
        h_angles_deg = [deg for deg in np.linspace(0, 360, h_steps, endpoint=False)]

        if distances is None:
            distances = self.distances

        assert min(distances) > self.target_diagonal/2, \
            f'minimum distance {min(distances)} less than half of target diagonal {self.target_diagonal/2}'
        assert max(v_angles_deg, key=abs) <= 75, \
            f'maximum vertical angle {max(v_angles_deg, key=abs)} greater than +/-75 [deg]'

        i:int = 0
        for distance in distances:
            for v_angle_deg in v_angles_deg:
                for h_angle_deg in h_angles_deg:

                    i += 1
                    id = f'{i:04d}'
                    self.render_orbit_view(h_angle_deg, v_angle_deg, distance, id)

        print(f"Rendered {i} images")

        self.write_intrinsics_to_file()
    
    def write_intrinsics_to_file(self):
        """
        Save camera intrinsics to a text file.
        """
        w, h, f = self.get_camera_intrinsics('mm')
        with open(os.path.join(self.render_dir, f'intrinsics.txt'), 'w') as file:
            file.write(f'{w} {h} {f}')


# TODO: incorporate query cx, cy intrinsics for rendering

# TODO: supress Blender terminal output

# TODO: run Blender code without task configuration
    # -> integrate with other files