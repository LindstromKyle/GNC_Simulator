import numpy as np

from environment import Environment
from vehicle import Vehicle


# Constants
G = 6.67430e-11
M_earth = 5.972e24
R_earth = 6371000

def test_gravity_at_surface():
    env = Environment()
    position = np.array([R_earth, 0, 0])  # 1 Earth radius away along x-axis
    mass = 1.0  # 1 kg test mass
    g_expected = -G * M_earth * mass / R_earth**2
    g_computed = env.gravitational_force(position, mass)
    assert np.isclose(np.linalg.norm(g_computed), abs(g_expected), rtol=1e-5)

def test_gravity_direction():
    env = Environment()
    position = np.array([0, R_earth, 0])
    force = env.gravitational_force(position, 1.0)
    direction = force / np.linalg.norm(force)
    assert np.allclose(direction, -position / np.linalg.norm(position), rtol=1e-5)

def test_drag_force_zero_velocity():
    env = Environment()
    vehicle = Vehicle()
    position = np.array([0, 0, R_earth + 1000])
    velocity = np.zeros(3)
    quaternion = np.array([1, 0, 0, 0])
    drag = env.drag_force(position, velocity, vehicle, quaternion)
    assert np.allclose(drag, np.zeros(3))

def test_drag_force_magnitude():
    env = Environment()
    vehicle = Vehicle()
    vehicle.drag_coefficient = 0.5
    vehicle.cross_sectional_area = 0.1
    velocity = np.array([0, 0, -100])  # 100 m/s downward
    position = np.array([0, 0, R_earth + 1000])
    quaternion = np.array([1, 0, 0, 0])
    drag = env.drag_force(position, velocity, vehicle, quaternion)
    rho = env.atmospheric_density(1000)
    expected_magnitude = 0.5 * rho * 100**2 * 0.5 * 0.1
    assert np.isclose(np.linalg.norm(drag), expected_magnitude, rtol=1e-3)