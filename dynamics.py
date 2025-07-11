

import numpy as np
from utils import compute_quaternion_derivative

def calculate_dynamics(time, state, vehicle, environment):
    position = state[0:3]
    velocity = state[3:6]
    quaternion = state[6:10]
    angular_velocity = state[10:13]

    # Mass
    vehicle_mass = vehicle.mass(time)

    # Forces
    thrust_force = vehicle.thrust_vector(time, quaternion)
    gravitational_force = environment.gravitational_force(position, vehicle_mass)
    drag_force = environment.drag_force(position, velocity, vehicle)
    net_force = thrust_force + gravitational_force + drag_force

    acceleration = net_force / vehicle_mass

    # Quaternion dynamics
    quaternion_derivative = compute_quaternion_derivative(quaternion, angular_velocity)

    # Angular dynamics
    moment_of_inertia = vehicle.moment_of_inertia
    torque = np.array([0, 0, 0]) # Start with no torque
    angular_momentum = moment_of_inertia @ angular_velocity
    gyroscopic_reaction_torque = np.cross(angular_velocity, angular_momentum)
    angular_acceleration = np.linalg.inv(moment_of_inertia) @ (torque - gyroscopic_reaction_torque)

    return np.concatenate([velocity, acceleration, quaternion_derivative, angular_acceleration])
