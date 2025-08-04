import numpy as np
import logging

from utils import compute_quaternion_derivative, rotate_vector_by_quaternion, compute_orbital_elements


def calculate_dynamics(time, state, vehicle, environment, log_flag, controls):
    position = state[0:3]
    velocity = state[3:6]
    quaternion = state[6:10]
    angular_velocity = state[10:13]
    propellant_mass = state[13]

    # if not np.isclose(np.linalg.norm(quaternion), 1):
    #     pass

    # if np.all(quaternion == np.array([0, 0, 0, 0])):
    #     pass

    # Mass
    vehicle_mass = vehicle.dry_mass + max(propellant_mass, 0)  # Clamp to avoid negative

    # Controller logic
    gimbal_angles = controls.get("engine_gimbal_angles")
    throttle = controls.get("throttle")
    desired_torque = controls.get("desired_torque")

    # Forces
    if throttle > 0 and propellant_mass > 0:
        thrust_force, thrust_vector_torque = vehicle.thrust_vector(
            time, quaternion, gimbal_angles, throttle, propellant_mass
        )
        mass_flow_rate = vehicle.mdot_max * throttle

    else:
        thrust_force = np.zeros(3)
        thrust_vector_torque = np.zeros(3)
        mass_flow_rate = 0.0

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
        logging.info(f"-----------------------------------[CONTROLLER]-------------------------------------------")
        logging.info(f"desired torque (N*m): {np.round(desired_torque, 4)} | throttle: {throttle:.4f}")
        logging.info(
            f"applied torque (N*m): {np.round(total_torque, 4)} | ang vel (rad/s): {np.round(angular_velocity, 4)} | ang acc (rad/s/s): {np.round(angular_acceleration, 4)}"
        )
        # TODO: fix discrepancy with 1 vs 9 engines (log parser will have to handle this?)
        if len(gimbal_angles) == 9:
            for i in range(3):
                logging.info(
                    f"gimbal angle {i+1} (deg): {np.round(np.rad2deg(gimbal_angles[i]), 8)} | gimbal angle {i+4} (deg): {np.round(np.rad2deg(gimbal_angles[i+3]), 8)} | gimbal angle {i+7} (deg): {np.round(np.rad2deg(gimbal_angles[i+6]), 8)}"
                )
        else:
            for i in range(3):
                logging.info(
                    f"gimbal angle 1 (deg): {np.round(np.rad2deg(gimbal_angles[0]), 8)} | gimbal angle 1 (deg): {np.round(np.rad2deg(gimbal_angles[0]), 8)} | gimbal angle 1 (deg): {np.round(np.rad2deg(gimbal_angles[0]), 8)}"
                )
        logging.info(f"------------------------------------[DYNAMICS]--------------------------------------------")
        logging.info(
            f"pos (m): {np.round(position, 4)} | vel (m/s): {np.round(velocity,4)} | acc (m/s/s): {np.round(acceleration, 4)}"
        )
        logging.info(
            f"thrust (N): {np.round(thrust_force, 4)} | drag (N): {np.round(drag_force, 4)} | gravity (N): {np.round(gravitational_force, 4)} | net force (N): {np.round(net_force, 4)}"
        )
        logging.info(
            f"total mass (kg): {vehicle_mass:.4f} | propellant mass (kg): {propellant_mass:.4f} | mass flow (kg/s): {mass_flow_rate:.4f}"
        )
        logging.info(f"")

    derivatives = np.concatenate(
        [velocity, acceleration, quaternion_derivative, angular_acceleration, [-mass_flow_rate]]
    )
    return derivatives
