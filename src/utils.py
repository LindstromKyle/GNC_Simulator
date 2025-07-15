import numpy as np
from scipy.spatial.transform import Rotation


def compute_acceleration(t_vals, velocity_vals):
    acceleration_vals = np.gradient(velocity_vals, t_vals)
    return acceleration_vals


def rotate_vector_by_quaternion(vector, quaternion):
    """
    Rotate a vector from body frame to ECI frame using a quaternion.
    Args:
        vector: 3D numpy array in body frame
        quaternion: Quaternion [w, x, y, z] representing ECI←body rotation

    returns: Rotated vector in ECI frame
    """
    # Reorder quaternion to match scipy's format: [q1, q2, q3, q0]
    q = np.array([quaternion[1], quaternion[2], quaternion[3], quaternion[0]])
    rotation = Rotation.from_quat(q)
    return rotation.apply(vector)

def compute_quaternion_derivative(quaternion, angular_velocity):
    """
    Computes dq/dt given quaternion and body-frame angular velocity.
    Quaternion is in [q0, q1, q2, q3] format.
    """
    w_x, w_y, w_z = angular_velocity
    Omega = np.array([
        [0, -w_x, -w_y, -w_z],
        [w_x, 0, w_z, -w_y],
        [w_y, -w_z, 0, w_x],
        [w_z, w_y, -w_x, 0]
    ])
    return 0.5 * Omega @ quaternion

def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def quaternion_inverse(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]]) / np.linalg.norm(q)**2

def quat_to_axis_angle(q: np.ndarray) -> np.ndarray:
    angle = 2 * np.arccos(q[0])
    if angle == 0:
        return np.array([1, 0, 0, 0])  # No rotation
    axis = q[1:] / np.sin(angle / 2)
    return np.append(axis, angle)