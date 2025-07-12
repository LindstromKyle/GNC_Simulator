import numpy as np
import logging

from utils import compute_quaternion_derivative

def calculate_dynamics(time, state, vehicle, environment, log):
    position = state[0:3]
    velocity = state[3:6]
    quaternion = state[6:10]
    angular_velocity = state[10:13]

    # Mass
    vehicle_mass = vehicle.mass(time)

    # Forces
    thrust_force = vehicle.thrust_vector(time, quaternion)
    gravitational_force = environment.gravitational_force(position, vehicle_mass)
    drag_force = environment.drag_force(position, velocity, vehicle, quaternion)
    net_force = thrust_force + gravitational_force + drag_force

    acceleration = net_force / vehicle_mass

    # Quaternion dynamics
    quaternion_derivative = compute_quaternion_derivative(quaternion, angular_velocity)

    # Angular dynamics
    moment_of_inertia = vehicle.moment_of_inertia
    thrust_vector_torque = np.array([0, 0, 0]) # Start with no torque
    aerodynamic_torque = environment.aerodynamic_torque(position, velocity, quaternion, angular_velocity, vehicle)
    total_torque = thrust_vector_torque + aerodynamic_torque
    angular_momentum = moment_of_inertia @ angular_velocity
    gyroscopic_reaction_torque = np.cross(angular_velocity, angular_momentum)
    angular_acceleration = np.linalg.inv(moment_of_inertia) @ (total_torque - gyroscopic_reaction_torque)

    # Log state evolution
    if log:
        logging.info(f"t={time:.2f}s | pos={position} | vel={velocity} | acc={acceleration}")
        logging.info(f"mass={vehicle_mass:.2f} | thrust={thrust_force} | drag={drag_force} | gravity={gravitational_force}")

    return np.concatenate([velocity, acceleration, quaternion_derivative, angular_acceleration])
