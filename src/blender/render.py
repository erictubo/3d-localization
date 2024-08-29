import bpy
import os
import numpy as np
from math import pi, sqrt, tan, asin, atan, atan2
from mathutils import Vector, Quaternion
from typing import Dict, Tuple, List, Optional, Union


class Blender:

    def __init__(self,
                 blend_file: str,
                 render_dir: str,
                 target_name: str = 'Model',
                 camera_name: str = 'Camera',
                 default_image_size: 'tuple[int]' = (1024, 1024),
                 default_focal_length_mm: float = 35,
                 depth_rendering: bool = True,
                 edge_rendering: bool = False,
                 images_prefix: str = 'images/',
                 depth_prefix: str = 'depth/',
                 poses_prefix: str = 'poses/',
                 intrinsics_prefix: str = 'intrinsics/',
                #  edges_prefix: str = 'edges/'
                 ):

        # Open blend file
        bpy.ops.wm.open_mainfile(filepath=blend_file)

        # Set directories
        if render_dir[-1] != '/':
            render_dir += '/'

        self.render_dir = render_dir

        print('Directory:' + self.render_dir)

        self.images_prefix = images_prefix
        self.depth_prefix = depth_prefix
        self.poses_prefix = poses_prefix
        # self.edges_prefix = edges_prefix
        self.images_dir = render_dir + images_prefix
        self.depth_dir = render_dir + depth_prefix
        self.poses_dir = render_dir + poses_prefix
        self.intrinsics_dir = render_dir + intrinsics_prefix
        # self.edges_dir = render_dir + edges_prefix

        for dir in [self.render_dir, self.images_dir, self.depth_dir, self.poses_dir, self.intrinsics_dir]:
            if not os.path.exists(dir): os.makedirs(dir)
        
        # Set objects
        self.target = bpy.data.objects[target_name]
        self.camera = bpy.data.objects[camera_name]
        self.light = bpy.data.objects['Light']

        # Set light to sun and strength to 10
        self.light.data.type = 'SUN'
        self.light.data.energy = 3

        # Set lens end clip to 1000
        self.camera.data.clip_end = 1000

        # Set rotation mode
        self.target.rotation_mode = 'QUATERNION'
        self.camera.rotation_mode = 'QUATERNION'
        self.light.rotation_mode = 'QUATERNION'

        self.get_target_dimensions()

        self.distances = self.calculate_distances()

        print("Distances:", self.distances)

        self.default_image_size = default_image_size
        self.default_focal_length_mm = default_focal_length_mm

        self.set_camera_intrinsics(default_image_size[0], default_image_size[1], default_focal_length_mm, 'MM')
        
        self.depth_rendering = depth_rendering
        if self.depth_rendering:
            self.prepare_depth_rendering()

        self.edge_rendering = edge_rendering
        if self.edge_rendering:
            self.prepare_edge_rendering()


    """
    GEOMETRY
    - target center
    - target size
    - target diagonal
    - bounding box corners
    - ground height
    """

    def get_target_dimensions(self):
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

        # Calculate 8 bounding box corners from center and size
        bbox_corners = [center + Vector([x, y, z]) for x in [-x_dim/2, x_dim/2] for y in [-y_dim/2, y_dim/2] for z in [-z_dim/2, z_dim/2]]
        
        size = Vector([x_dim, y_dim, z_dim])
        diagonal = sqrt(x_dim**2 + y_dim**2 + z_dim**2)
        ground_height = center.z - size.z/2
        
        assert diagonal != 0, "Diagonal is zero"

        self.target_center = center
        self.target_size = size
        self.target_diagonal = diagonal
        self.bbox_corners = bbox_corners
        self.ground_height = ground_height

        print(f"Target center: {self.target_center}")
        print(f"Target size: {self.target_size}")
        print(f"Target diagonal: {self.target_diagonal}")
        print(f"Bounding box corners: {self.bbox_corners}")
        print(f"Ground height: {self.ground_height}")


    def write_bounding_box_to_file(self):
        """
        Save bounding box corners to a text file.
        """
        with open(os.path.join(self.render_dir, f'bounding_box.txt'), 'w') as file:
            for corner in self.bbox_corners:
                file.write(f'{corner.x} {corner.y} {corner.z}\n')
    
    def calculate_distances(self) -> List[int]:
        """
        Calculate distance values for rendering based on target size and diagonal.
        """
        d = np.ceil(self.target_diagonal/2/10)*10

        return [d, 2*d]
    
    def convert_orbit_to_pose(self, distance: float, h_angle: float, v_angle: float, unit: str = 'rad') -> np.ndarray:
        """
        Convert orbit view parameters to camera pose.
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
        px, py, pz = self.target_center + rotated_offset

        direction = self.target_center - Vector((px, py, pz))
        direction.normalize()
        qw, qx, qy, qz = direction.to_track_quat('-Z', 'Y')

        return np.array([px, py, pz, qw, qx, qy, qz])
    
    def convert_pose_to_orbit(self, pose: np.ndarray) -> Tuple[float, float, float]:
        """
        Convert camera pose to orbit view parameters.
        """
        px, py, pz, qw, qx, qy, qz = pose

        distance = sqrt((px - self.target_center.x)**2 + (py - self.target_center.y)**2 + (pz - self.target_center.z)**2)
        x, y, z = self.target_center - Vector((px, py, pz))

        h_angle = atan2(y, x)
        v_angle = asin(z/distance)
        
        return distance, h_angle, v_angle


    """
    CAMERA
    - intrinsics
    - pose
    - orbit pose
    """

    def get_camera_intrinsics(self, f_unit: str) -> 'tuple[int, int, float, float, float]':
        """
        Get camera intrinsics (w, h, f, cx, cy)
        """
        w = bpy.context.scene.render.resolution_x
        h = bpy.context.scene.render.resolution_y

        cx = w/2 - max(w,h) * self.camera.data.shift_x
        cy = h/2 + max(w,h) * self.camera.data.shift_y

        fov_rad = self.camera.data.angle
        fov_deg = fov_rad * 180/pi

        f_pix = max(w, h) / (2 * tan(fov_rad/2))

        f_mm = self.camera.data.lens

        if f_unit.upper() == 'MM':
            f = f_mm
        elif f_unit.upper() == 'PIX':
            f = f_pix
        elif f_unit.upper() == 'DEG':
            f = fov_deg
        elif f_unit.upper() == 'RAD':
            f = fov_rad
        else:
            raise ValueError(f"Select valid focal length unit: 'MM', 'PIX', 'DEG' or 'RAD'")

        return w, h, f, f_unit, cx, cy


    def set_camera_intrinsics(self, w: int, h: int, f: float, f_unit: str, cx: float = None, cy: float = None):
        """
        Set camera intrinsics (focal_length, w, h).
        """
        bpy.context.scene.render.resolution_x = w
        bpy.context.scene.render.resolution_y = h

        if f_unit.upper() == 'MM':
            self.camera.data.lens = f
        elif f_unit.upper() == 'PIX':
            self.camera.data.angle = 2 * atan(max(w, h)/(2*f))
        elif f_unit.upper() == 'DEG':
            self.camera.data.angle = f * pi/180
        elif f_unit.upper() == 'RAD':
            self.camera.data.angle = f
        else:
            raise ValueError(f"Select valid focal length unit: 'MM', 'PIX', 'DEG' or 'RAD'")
        
        if cx is not None:
            self.camera.data.shift_x = (w/2 - cx)/max(w, h)
        else:
            self.camera.data.shift_x = 0
        
        if cy is not None:
            self.camera.data.shift_y = (cy - h/2)/max(w, h)
        else:
            self.camera.data.shift_y = 0
        
    def write_camera_intrinsics(self, id: str, f_unit: str = 'MM'):
        """
        Save camera intrinsics to a text file.
        """
        w, h, f, f_unit, cx, cy = self.get_camera_intrinsics(f_unit)
        with open(os.path.join(self.intrinsics_dir, f'{id}.txt'), 'w') as file:
            file.write(f'{w} {h} {f} {f_unit} {cx} {cy}')


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
        self.camera.rotation_quaternion = q @ self.camera.rotation_quaternion

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

    def set_camera_orbit_pose(self, distance: float, h_angle: float, v_angle: float, unit: str = 'rad', lighting: bool = True):
        """
        Set camera pose in orbit view around target object.
        """
        pose = self.convert_orbit_to_pose(distance, h_angle, v_angle, unit)
        self.set_camera_pose(pose)
        
        # if unit == 'deg':
        #     h_angle *= pi/180
        #     v_angle *= pi/180
    
        # h_axis = Vector((0, 0, 1))
        # h_quat = Quaternion(h_axis, h_angle)

        # v_axis = Vector((0, -1, 0))
        # v_quat = Quaternion(v_axis, v_angle)

        # combined_quat = h_quat @ v_quat

        # offset = Vector((distance, 0, 0))
        # rotated_offset = combined_quat @ offset
        # self.camera.location = self.target_center + rotated_offset

        # direction = self.target_center - self.camera.location
        # direction.normalize()
        # quat_to_target = direction.to_track_quat('-Z', 'Y')
        # self.camera.rotation_quaternion = quat_to_target

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
            relative_h_angle *= pi/180
            relative_v_angle *= pi/180
        
        distance, h_angle, v_angle = self.get_camera_orbit_pose()

        distance += relative_distance
        h_angle += relative_h_angle
        v_angle += relative_v_angle

        self.set_camera_orbit_pose(distance, h_angle, v_angle)

    """
    LIGHTING
    """

    def get_lighting_pose(self) -> np.ndarray:
        """
        Get lighting pose (px, py, pz, qx, qy, qz, qw).
        """
        px, py, pz = self.light.location
        qw, qx, qy, qz = self.light.rotation_quaternion
        pose = np.array([px, py, pz, qw, qx, qy, qz])

        return pose

    def set_lighting_pose(self, pose: np.ndarray):
        """
        Set lighting pose (px, py, pz, qw, qx, qy, qz) in world coordinates.
        """
        px, py, pz, qw, qx, qy, qz = pose

        self.light.location = Vector((px, py, pz))
        self.light.rotation_quaternion = Quaternion((qw, qx, qy, qz))

    def set_lighting_orbit_pose(self, distance: float, h_angle: float, v_angle: float, unit: str = 'rad'):
        """
        Set lighting pose in orbit view around target object.
        """
        pose = self.convert_orbit_to_pose(distance, h_angle, v_angle, unit)
        self.set_lighting_pose(pose)


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
    
    def render(self, id: str):
        """
        Render images (+ depth and edges if enabled), save camera pose and depth values.
        """
        image_file = os.path.join(self.images_dir, f'{id}.png')

        if os.path.exists(image_file):
            print(f"Render {id} already exists, skipping")
            return
        
        bpy.context.scene.render.filepath = image_file

        if self.edge_rendering:
            self.edges_output.file_slots[0].path = id
        
        self.depth_output.file_slots[0].path = id

        bpy.ops.render.render(write_still=True)

        self.write_camera_pose(id)
        self.write_camera_intrinsics(id)

    def render_orbit_view(self, id: str, h_angle_deg: int, v_angle_deg: int, distance: int, f: float = 35, f_unit: str = 'MM'):
        """
        Render orbit view at a specific distance, horizontal and vertical angle (deg).
        """
        h_angle = h_angle_deg * pi/180
        v_angle = v_angle_deg * pi/180

        assert distance > self.target_diagonal/2, \
            f' distance {distance} not consistenly outside the target bounding box diagonal {self.target_diagonal/2}'
        assert abs(v_angle_deg) <= 75, \
            f'vertical angle {v_angle} beyond +/-75 [deg]'

        self.set_camera_orbit_pose(distance, h_angle, v_angle)
        self.set_camera_intrinsics(self.default_image_size[0], self.default_image_size[1], f, f_unit)

        assert self.camera.location.z >= (self.target_center.z - self.target_size.z/2), \
            f'camera z position {self.camera.location.z} below the ground {self.target_center.z - self.target_size.z/2}'
        
        self.render(id)

    def render_orbit_views(self, h_steps: int, v_angles_deg: 'list[int]', distances: 'list[int]' = None, focal_lengths: 'list[float]' = [35], f_unit: str = 'MM'):
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
        
        total = len(focal_lengths) * len(distances) * len(h_angles_deg) * len(v_angles_deg)

        print(f"Rendering {total} images ...")

        i:int = 0

        for f in focal_lengths:
            for distance in distances:
                for v_angle_deg in v_angles_deg:
                    for h_angle_deg in h_angles_deg:

                        i += 1
                        id = f'{i:04d}'
                        id = f'f{int(f)}_d{int(distance)}_v{int(v_angle_deg)}_h{int(h_angle_deg)}'

                        # skip if already exists
                        if os.path.exists(os.path.join(self.images_dir, f'{id}.png')):
                            print(f"Render {id} already exists, skipping")
                            continue

                        self.render_orbit_view(id, h_angle_deg, v_angle_deg, distance, f, f_unit)

                        print(f"Render {i} / {total} ...")

        self.write_bounding_box_to_file()

    def render_ground_view(self, id: str, ground_distance: int, h_angle_deg: int, height_above_ground: int, f: float = 35, f_unit: str = 'MM', v_angle_deg: int = 0):
        """
        Render ground view at a specific distance, horizontal angle (deg) and height.
        """
        assert height_above_ground >= 0, \
            f'height {height_above_ground} below ground level'
        
        h_angle = h_angle_deg * pi/180
        height = self.ground_height + height_above_ground
        
        v_angle = atan2(height - self.target_center.z, ground_distance)

        distance = sqrt(ground_distance**2 + (self.target_center.z - height)**2)
        assert distance > self.target_diagonal/2, \
            f' distance {distance} not consistenly outside the target bounding box diagonal {self.target_diagonal/2}'

        self.set_camera_orbit_pose(distance, h_angle, v_angle)

        if v_angle_deg != 0:
            # rotate camera upwards by v_angle_deg
            axis = Vector((0, 0, 1)).cross(self.camera.location - self.target_center)
            q = Quaternion(axis, v_angle_deg * pi/180)
            self.adjust_camera_pose([0, 0, 0, q.w, q.x, q.y, q.z])

        self.set_camera_intrinsics(self.default_image_size[0], self.default_image_size[1], f, f_unit)

        self.set_lighting_orbit_pose(distance, h_angle, (pi/2 + v_angle)/2)
        
        self.render(id)

    def render_ground_views(self, distances: 'list[int]', h_steps: int, heights: 'list[int]', focal_lengths: 'list[float]' = [35], f_unit: str = 'MM', v_angles_deg: 'list[int]' = [0]):
        """
        Render ground views at multiple distances, horizontal angles (deg) and heights.
        """
        assert min(distances) > self.target_diagonal/2, \
            f'minimum distance {min(distances)} less than half of target diagonal {self.target_diagonal/2}'
        assert min(heights) >= 0, \
            f'minimum height {min(heights)} below ground level'
        
        h_angles_deg = [deg for deg in np.linspace(0, 360, h_steps, endpoint=False)]

        total = len(distances) * len(h_angles_deg) * len(heights)

        print(f"Rendering {total} images ...")

        i:int = 0
        for f in focal_lengths:
            for distance in distances:
                for h_angle_deg in h_angles_deg:
                    for height in heights:
                        for v_angle_deg in v_angles_deg:

                            i += 1
                            id = f'{i:04d}'
                            id = f'f{int(f)}_d{int(distance)}_z{int(height)}_h{int(h_angle_deg)}'

                            if v_angle_deg != 0:
                                # add vertical angle to id, '+' for positive angles and '-' for negative angles
                                if v_angle_deg < 0:
                                    id += f'_v{v_angle_deg}'
                                else:
                                    id += f'_v+{v_angle_deg}'


                            # skip if already exists
                            if os.path.exists(os.path.join(self.images_dir, f'{id}.png')):
                                print(f"Render {id} already exists, skipping")
                                continue

                            self.render_ground_view(id, distance, h_angle_deg, height, f, f_unit, v_angle_deg)

                            print(f"Render {i} / {total} ...")

        self.write_bounding_box_to_file()


if __name__ == "__main__":
    blender_dir = '/Users/eric/Library/Mobile Documents/com~apple~CloudDocs/Blender/'
    blend_file = f'{blender_dir}assets/models/notre dame E/notre dame E.blend'
    target_name = 'notre-dame-de-paris-complete-miniworld3d'

    render_dir = f'{blender_dir}renders/notre_dame_E_lighting/'

    blender = Blender(
        blend_file,
        render_dir,
        target_name,
        )

    blender.render_ground_views(
        distances=[110],
        h_steps = 8,
        heights = [10]
        )