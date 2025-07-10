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