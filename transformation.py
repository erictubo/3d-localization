import numpy as np
from math import pi
import typing
from scipy.spatial.transform import Rotation
from pyquaternion import Quaternion


def pose_to_matrix(pose: np.ndarray) -> np.ndarray:
    """
    Convert pose vector to transformation matrix.
    Format: scalar-first (px, py, pz, qw, qx, qy, qz)
    """
    assert pose.shape == (7,), pose.shape
    t = pose[:3]
    q = Quaternion(pose[3:])
    R = q.rotation_matrix
    T = np.eye(4)
    T[0:3, :] = np.c_[R, t]

    return T

def matrix_to_pose(T: np.ndarray) -> np.ndarray:
    """
    Convert transformation matrix to pose vector.
    Format: scalar-first (px, py, pz, qw, qx, qy, qz)
    """
    assert T.shape == (4,4), T.shape
    t = T[:3, 3]
    assert t.shape == (3,), t.shape
    R = T[:3, :3]
    assert R.shape == (3,3), R.shape
    q = Quaternion(matrix=R).q
    pose = np.append(t, q)

    return pose

def invert_pose(pose: np.ndarray) -> np.ndarray:
    """
    Invert pose vector.
    """
    assert pose.shape == (7,), pose.shape
    t = pose[:3]
    q = Quaternion(pose[3:])
    q_inv = q.inverse
    t_inv = -Quaternion(q).rotate(t)
    pose_inv = np.append(t_inv, q_inv.q)

    return pose_inv

def invert_matrix(T: np.ndarray) -> np.ndarray:
    """
    Invert transformation matrix.
    Only for orthogonal matrices.
    """
    # assert that rotation matrix is orthogonal
    assert np.linalg.inv(T[:3, :3]) == T[:3, :3].T
    assert T.shape == (4,4), T.shape
    R = T[:3, :3]
    t = T[:3, 3]
    R_inv = Rotation.from_matrix(R).inv().as_matrix()
    t_inv = -R_inv @ t
    T_inv = np.eye(4)
    T_inv[0:3, :]  = np.c_[R_inv, t_inv]
    assert T_inv.shape == (4,4), T_inv.shape

    return T_inv

# def rotate_quaternion(q: np.ndarray, angle_deg: float, axis: str) -> np.ndarray:
#     """
#     Rotate quaternion around axis.
#     """
#     assert q.shape == (4,), q.shape
#     assert axis in ['x', 'y', 'z'], axis
#     angle = angle_deg * pi / 180
#     initial = Rotation.from_quat([q[1], q[2], q[3], q[0]]) # scalar-last in scipy
#     rotation = Rotation.from_rotvec(angle * np.array([1, 0, 0])) if axis == 'x' \
#         else Rotation.from_rotvec(angle * np.array([0, 1, 0])) if axis == 'y' \
#         else Rotation.from_rotvec(angle * np.array([0, 0, 1])) if axis == 'z' \
#         else None
#     q_new_xyzw = (initial * rotation).as_quat() # scalar-last in scipy
#     q_new = np.append(q_new_xyzw[3], q_new_xyzw[:3]) # scalar-first

#     return q_new

# def rotate_pose(pose: np.ndarray, angle_deg: float, axis: str) -> np.ndarray:
#     """
#     Rotate pose vector around axis.
#     """
#     assert pose.shape == (7,), pose.shape
#     assert axis in ['x', 'y', 'z'], axis
#     t = pose[:3]
#     assert t.shape == (3,), t.shape
#     q = pose[3:]
#     assert q.shape == (4,), q.shape
#     angle = angle_deg * pi / 180
#     R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix() # scalar-last in scipy
#     R_rot = Rotation.from_rotvec(angle * np.array([1, 0, 0])).as_matrix() if axis == 'x' \
#         else Rotation.from_rotvec(angle * np.array([0, 1, 0])).as_matrix() if axis == 'y' \
#         else Rotation.from_rotvec(angle * np.array([0, 0, 1])).as_matrix() if axis == 'z' \
#         else None
#     R_new = R @ R_rot
#     q_xyzw = Rotation.from_matrix(R_new).as_quat() # scalar-last in scipy
#     q = np.append(q_xyzw[3], q_xyzw[:3]) # scalar-first
#     pose_rot = np.append(t, q)

#     return pose_rot


# def scale_pose(pose: np.ndarray, scale: float) -> np.ndarray:
#     """
#     Scale pose vector.
#     """
#     assert pose.shape == (7,), pose.shape
#     t = pose[:3]
#     assert t.shape == (3,), t.shape
#     q = pose[3:]
#     assert q.shape == (4,), q.shape
#     t_scaled = scale * t
#     pose_scaled = np.append(t_scaled, q)

#     return pose_scaled


# class Quaternion:
#     def __init__(self, w=1, x=0, y=0, z=0):
#         self.w = w
#         self.x = x
#         self.y = y
#         self.z = z

#     def __str__(self) -> str:
#         return f'w: {self.w}, x: {self.x}, y: {self.y}, z: {self.z}'
    
#     def __mul__(self, other: 'Quaternion') -> 'Quaternion':
#         w = self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z
#         x = self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y
#         y = self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x
#         z = self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w
#         return Quaternion(w, x, y, z)
    


# class Translation:
#     def __init__(self, x=0, y=0, z=0):
#         self.x = x
#         self.y = y
#         self.z = z
    
#     def __str__(self) -> str:
#         return f'x: {self.x}, y: {self.y}, z: {self.z}'
    
#     def __add__(self, other: 'Translation') -> 'Translation':
#         x = self.x + other.x
#         y = self.y + other.y
#         z = self.z + other.z
#         return Translation(x, y, z)
    
    