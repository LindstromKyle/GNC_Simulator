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


def compute_minimal_quaternion_rotation(desired_z_vector):
    # Compute minimal quaternion to rotate body Z [0,0,1] to desired_z
    body_z_vector = np.array([0, 0, 1])
    dot = np.dot(body_z_vector, desired_z_vector)
    dot = np.clip(dot, -1.0, 1.0)

    if dot > 0.99999:
        return np.array([1.0, 0.0, 0.0, 0.0])
    if dot < -0.99999:
        # 180° flip (arbitrary axis)
        return np.array([0.0, 1.0, 0.0, 0.0])

    cross_product = np.cross(body_z_vector, desired_z_vector)
    cross_norm = np.linalg.norm(cross_product)
    if cross_norm < 1e-6:
        axis = np.array([0.0, 0.0, 0.0])
    else:
        axis = cross_product / cross_norm
    angle = np.arccos(dot)
    sin_half = np.sin(angle / 2.0)
    cos_half = np.cos(angle / 2.0)
    quat = np.array([cos_half, sin_half * axis[0], sin_half * axis[1], sin_half * axis[2]])
    return quat / np.linalg.norm(quat)


def compute_orbital_elements(position: np.ndarray, velocity: np.ndarray, mu: float) -> dict | None:
    """
    Compute Keplerian elements from position and velocity (inertial frame).
    Returns dict with 'a', 'e', 'ra', 'rp', 'e_vec' or None if hyperbolic.
    """
    r = np.linalg.norm(position)
    v2 = np.dot(velocity, velocity)
    energy = v2 / 2 - mu / r
    if energy >= 0:
        return None  # Hyperbolic or parabolic - no bounded apo/peri
    a = -mu / (2 * energy)
    h = np.cross(position, velocity)
    e_vec = (1 / mu) * np.cross(velocity, h) - position / r
    e = np.linalg.norm(e_vec)
    ra = a * (1 + e)
    rp = a * (1 - e)
    return {"a": a, "e": e, "ra": ra, "rp": rp, "e_vec": e_vec}


def compute_time_to_apoapsis(position: np.ndarray, velocity: np.ndarray, elements: dict, mu: float) -> float:
    """
    Compute time (s) to next apoapsis from current state.
    Assumes elliptic orbit (elements not None).
    """
    r = np.linalg.norm(position)
    e_vec = elements["e_vec"]
    e = elements["e"]
    a = elements["a"]
    # True anomaly nu
    cos_nu = np.dot(e_vec, position) / (e * r)
    cos_nu = np.clip(cos_nu, -1.0, 1.0)
    nu0 = np.arccos(cos_nu)
    vr = np.dot(velocity, position) / r  # Radial velocity
    if vr < 0:
        nu = 2 * np.pi - nu0
    else:
        nu = nu0
    # Eccentric anomaly E
    sqrt_term = np.sqrt((1 - e) / (1 + e))
    tan_half_nu = np.tan(nu / 2)
    E = 2 * np.arctan(sqrt_term * tan_half_nu)
    if E < 0:
        E += 2 * np.pi
    # Mean anomaly M
    M = E - e * np.sin(E)
    # Delta M to next apoapsis (M = pi at apo)
    if M < np.pi:
        delta_M = np.pi - M
    else:
        delta_M = np.pi - M + 2 * np.pi
    # Mean motion n
    n = np.sqrt(mu / a**3)
    time_to_apo = delta_M / n
    return time_to_apo
