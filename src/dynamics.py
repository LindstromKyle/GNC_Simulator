import numpy as np
import logging

from utils import compute_quaternion_derivative, rotate_vector_by_quaternion


def calculate_dynamics(time, state, vehicle, environment, log_flag, controller=None):
    position = state[0:3]
    velocity = state[3:6]
    quaternion = state[6:10]
    angular_velocity = state[10:13]
    propellant_mass = state[13]

    # Mass
    vehicle_mass = vehicle.dry_mass + max(propellant_mass, 0)  # Clamp to avoid negative

    # Controller logic
    if controller:
        controls = controller.update(time, state)
        gimbal_angles = controls.get("engine_gimbal_angles", np.zeros(2))
        throttle = controls.get("throttle", 1.0)
    else:
        gimbal_angles = np.zeros(2)
        throttle = 1.0

    # Forces
    if throttle > 0 and propellant_mass > 0:
        thrust_force, thrust_vector_torque = vehicle.thrust_vector(time, quaternion, gimbal_angles, throttle)
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
        logging.info(
            f"t={time:.2f}s | pos={np.round(position, 2)} | vel={np.round(velocity,2)} | acc={np.round(acceleration, 2)}"
        )
        logging.info(
            f"thrust={np.round(thrust_force, 2)} | drag={np.round(drag_force, 2)} | gravity={np.round(gravitational_force, 2)} | net force={np.round(net_force, 2)}"
        )
        logging.info(f"quat={quaternion} | attitude(Z)={rotate_vector_by_quaternion(np.array([0,0,1]), quaternion)}")
        logging.info(
            f"total torque={np.round(total_torque, 2)} | ang vel={np.round(angular_velocity, 2)} | ang acc={np.round(angular_acceleration, 2)}"
        )
        logging.info(
            f"total mass={vehicle_mass:.2f} | propellant mass={propellant_mass:.2f} | mass flow={mass_flow_rate}"
        )
        logging.info(f"")

    derivatives = np.concatenate(
        [velocity, acceleration, quaternion_derivative, angular_acceleration, [-mass_flow_rate]]
    )
    return derivatives
