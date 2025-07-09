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
    # Reorder quaternion to match scipy's format: [x, y, z, w]
    q = np.array([quaternion[1], quaternion[2], quaternion[3], quaternion[0]])
    rotation = Rotation.from_quat(q)
    return rotation.apply(vector)