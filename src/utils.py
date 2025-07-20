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
    return np.round(rotation.apply(vector), 4)


def get_rotated_basis_from_quat(quaternion, basis=None):
    """
    Compute the rotated body-frame basis vectors in ECI frame.
    Args:
        quaternion: Quaternion [w, x, y, z] representing ECI←body rotation
        basis: Optional param - basis to start rotation from

    Returns: Dict with rotated unit vectors for body X, Y, Z in ECI
    """
    # Reorder quaternion to match scipy's format: [q1, q2, q3, q0]
    q = np.array([quaternion[1], quaternion[2], quaternion[3], quaternion[0]])
    rotation = Rotation.from_quat(q)

    if basis:
        rotated_basis = {
            "X": np.round(rotation.apply(basis["X"]), 4),
            "Y": np.round(rotation.apply(basis["Y"]), 4),
            "Z": np.round(rotation.apply(basis["Z"]), 4),
        }

    else:
        rotated_basis = {
            "X": np.round(rotation.apply([1, 0, 0]), 4),
            "Y": np.round(rotation.apply([0, 1, 0]), 4),
            "Z": np.round(rotation.apply([0, 0, 1]), 4),
        }

    return rotated_basis


def compute_quaternion_derivative(quaternion, angular_velocity):
    """
    Computes dq/dt given quaternion and body-frame angular velocity.
    Quaternion is in [q0, q1, q2, q3] format.
    """
    w_x, w_y, w_z = angular_velocity
    Omega = np.array(
        [
            [0, -w_x, -w_y, -w_z],
            [w_x, 0, w_z, -w_y],
            [w_y, -w_z, 0, w_x],
            [w_z, w_y, -w_x, 0],
        ]
    )
    return 0.5 * Omega @ quaternion


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def quaternion_inverse(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]]) / np.linalg.norm(q) ** 2


def quat_to_angle_axis(q: np.ndarray) -> np.ndarray:
    angle = 2 * np.arccos(q[0])
    if angle == 0:
        return np.array([0, 0, 0, 0])  # No rotation
    axis = q[1:] / np.sin(angle / 2)
    return np.append(angle, axis)


def angle_axis_to_quat(angle_axis: np.ndarray) -> np.ndarray:
    axis_norm = np.linalg.norm(angle_axis[1:])
    angle = angle_axis[0]
    if (axis_norm == 0) or (angle == 0):
        return np.array([1, 0, 0, 0])
    unit_axis = angle_axis[1:] / axis_norm
    quaternion = np.append(np.array([np.cos(angle / 2)]), np.sin(angle / 2) * unit_axis)
    return quaternion / np.linalg.norm(quaternion)
