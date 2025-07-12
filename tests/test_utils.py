import numpy as np

from utils import rotate_vector_by_quaternion

def test_rotate_vector_by_quaternion_identity():
    v = np.array([1, 0, 0])
    q_identity = np.array([1, 0, 0, 0])  # No rotation
    rotated = rotate_vector_by_quaternion(v, q_identity)
    assert np.allclose(rotated, v)