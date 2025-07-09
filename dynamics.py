

import numpy as np

def calculate_dynamics(time, state, vehicle, environment):
    position = state[0:3]
    velocity = state[3:6]
    quaternion = state[6:10]
    omega = state[10:13]

    # Mass
    vehicle_mass = vehicle.mass(time)

    # Forces
    thrust_force = vehicle.thrust_vector(time, quaternion)
    gravitational_force = environment.gravitational_force(position, vehicle_mass)
    drag_force = environment.drag_force(position, velocity, vehicle)

    net_force = thrust_force + gravitational_force + drag_force

    acceleration = net_force / vehicle_mass

    # Quaternion dynamics
    w_x, w_y, w_z = omega
    Omega = np.array([
        [0, -w_x, -w_y, -w_z],
        [w_x, 0, w_z, -w_y],
        [w_y, -w_z, 0, w_x],
        [w_z, w_y, -w_x, 0]
    ])
    dqdt = 0.5 * Omega @ quaternion

    # Angular dynamics
    moment_of_inertia = vehicle.inertia
    torque = np.array([0, 0, 0])  # Start with no torque
    domega = np.linalg.inv(moment_of_inertia) @ (torque - np.cross(omega, moment_of_inertia @ omega))

    return np.concatenate([velocity, acceleration, dqdt, domega])
