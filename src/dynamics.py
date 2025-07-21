import numpy as np
import logging

from utils import compute_quaternion_derivative, rotate_vector_by_quaternion


def calculate_dynamics(time, state, vehicle, environment, log_flag, controller=None):
    position = state[0:3]
    velocity = state[3:6]
    quaternion = state[6:10]
    angular_velocity = state[10:13]

    # Mass
    vehicle_mass = vehicle.mass(time)

    # Controller logic
    if controller:
        controls = controller.update(time, state)
        gimbal_angles = controls.get("engine_gimbal_angles", np.zeros(2))
        throttle = controls.get("throttle", 1.0)
    else:
        gimbal_angles = np.zeros(2)
        throttle = 1.0

    # Forces
    thrust_force, thrust_vector_torque = vehicle.thrust_vector(time, quaternion, gimbal_angles, throttle)
    gravitational_force = environment.gravitational_force(position, vehicle_mass)
    drag_force = environment.drag_force(position, velocity, vehicle, quaternion)
    net_force = thrust_force + gravitational_force + drag_force

    acceleration = net_force / vehicle_mass

    # Quaternion dynamics
    quaternion_derivative = compute_quaternion_derivative(quaternion, angular_velocity)

    # Angular dynamics
    moment_of_inertia = vehicle.moment_of_inertia
    aerodynamic_torque = environment.aerodynamic_torque(position, velocity, quaternion, angular_velocity, vehicle)
    total_torque = thrust_vector_torque + aerodynamic_torque
    angular_momentum = moment_of_inertia @ angular_velocity
    gyroscopic_reaction_torque = np.cross(angular_velocity, angular_momentum)
    angular_acceleration = np.linalg.inv(moment_of_inertia) @ (total_torque - gyroscopic_reaction_torque)

    # Log state evolution
    if log_flag:
        logging.info(f"t={time:.2f}s | pos={position} | vel={velocity} | acc={acceleration}")
        logging.info(
            f"thrust={thrust_force} | drag={drag_force} | gravity={gravitational_force} | net force={net_force}"
        )
        logging.info(f"quat={quaternion} | attitude(Z)={rotate_vector_by_quaternion(np.array([0,0,1]), quaternion)}")
        logging.info(
            f"total torque={total_torque} | angular_velocity={angular_velocity} | ang_accel={angular_acceleration}"
        )
        logging.info(f"mass={vehicle_mass:.2f}")
        logging.info(f"")

    derivatives = np.concatenate([velocity, acceleration, quaternion_derivative, angular_acceleration])
    return derivatives
