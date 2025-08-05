import logging

import numpy as np
from abc import ABC, abstractmethod

from guidance import Guidance
from state import State
from utils import quaternion_multiply, quaternion_inverse, quat_to_angle_axis, rotate_vector_by_quaternion
from vehicle import Vehicle


class Controller(ABC):
    """
    Base class for controllers. Subclasses implement control logic.
    """

    @abstractmethod
    def update(self, time: float, state: State, mission_planner_setpoints: dict, log_flag: bool) -> dict:
        """
        Compute control inputs based on time and current state.

        Args:
            log_flag ():
            mission_planner_setpoints ():
            time: Current simulation time (s)
            state: Current State object

        Returns:
            Dict with control outputs, e.g., {'gimbal_angles': np.array([pitch, yaw]), 'fin_deflections': dict(...)}
        """
        pass


class PIDAttitudeController(Controller):
    def __init__(
        self,
        kp: np.ndarray,
        ki: np.ndarray,
        kd: np.ndarray,
        guidance: Guidance,
        vehicle: Vehicle,
    ):
        """

        Args:
            kp ():
            ki ():
            kd ():
            guidance ():
            vehicle ():
        """
        self.kp = kp
        self.ki = ki.copy()  # Allow modification
        self.kd = kd
        self.guidance = guidance
        self.integral_error = np.zeros(3)  # Accumulator for I term
        self.previous_error = np.zeros(3)
        self.previous_d_term = np.zeros(3)
        self.vehicle = vehicle
        self.last_update_time = None

        # TODO: get rid of this when roll control is ready for stage 2
        # Handle uncontrollable roll for single-engine vehicles (e.g., Stage 2)
        if self.vehicle.num_engines == 1:
            self.ki[2] = 0.0  # Disable integral for roll (no windup)
            # Optional: Reduce or disable P/D for roll if desired
            # self.kp[2] *= 0.1  # Mild damping only
            # self.kd[2] *= 0.1

    def update(self, time: float, state_vector: np.ndarray, mission_planner_setpoints: dict, log_flag: bool) -> dict:
        current_quaternion = state_vector[6:10]
        current_propellant_mass = state_vector[13]

        # Throttle
        throttle = mission_planner_setpoints.get("throttle", 1.0)

        # Get desired quaternion from guidance
        desired_quaternion = self.guidance.get_desired_quaternion(time, state_vector, mission_planner_setpoints)
        desired_quaternion /= np.linalg.norm(desired_quaternion)

        # Compute quaternion error (expressed in Body basis vectors)
        error_quaternion = quaternion_multiply(desired_quaternion, quaternion_inverse(current_quaternion))
        error_quaternion /= np.linalg.norm(error_quaternion)

        # Convert to angle-axis for PID
        angle_axis = quat_to_angle_axis(error_quaternion)
        current_error = angle_axis[0] * angle_axis[1:]  # angle (rad) * Axis

        # Basic gain scheduling
        # Example: Scale kp/kd inversely with mass (higher control authority as mass drops)
        current_mass = self.vehicle.dry_mass + current_propellant_mass
        # TODO: think about this
        # mass_ratio = current_mass / self.vehicle.dry_mass  # >1 early, ~1 late
        # kp_scheduled = self.kp / mass_ratio  # Lower early, higher late
        # kd_scheduled = self.kd / mass_ratio**0.5  # Mild scaling
        kp_scheduled = self.kp
        kd_scheduled = self.kd

        # Compute PID terms
        d_term = np.zeros(3)
        if self.last_update_time:
            dt = time - self.last_update_time
            if dt > 0:
                self.integral_error += current_error * dt
                d_term = kd_scheduled * (current_error - self.previous_error) / dt

        # Low pass filter on d term
        # TODO: put alpha in __init__?
        # alpha = 0.1
        # d_term = alpha * d_term + (1 - alpha) * self.previous_d_term
        # self.previous_d_term = d_term

        i_term = self.ki * self.integral_error
        p_term = kp_scheduled * current_error

        self.last_update_time = time
        self.previous_error = current_error

        # Compute unsaturated torque
        unsaturated_torque = p_term + i_term + d_term

        # Clip torque for saturation (anti-windup prep)
        control_torque = unsaturated_torque.copy()

        # Map torque to actuators
        thrust_magnitude = self.vehicle.get_thrust_magnitude(time)
        effective_thrust_magnitude = thrust_magnitude * throttle
        gimbal_arm = self.vehicle.get_gimbal_arm(current_propellant_mass)
        max_torque = effective_thrust_magnitude * np.sin(self.vehicle.engine_gimbal_limit_rad) * gimbal_arm
        # Clamp control torque to max for pitch and yaw
        control_torque[0] = np.clip(control_torque[0], -max_torque, max_torque)
        control_torque[1] = np.clip(control_torque[1], -max_torque, max_torque)

        # TODO: get rid of this when roll control implemented for stage 2
        if self.vehicle.num_engines == 1:
            control_torque[2] = 0.0

        # Anti-windup: Recalculate integral error based on saturated torque
        # Skip axes where ki=0 to avoid division by zero
        mask = self.ki != 0
        self.integral_error[mask] = (control_torque[mask] - p_term[mask] - d_term[mask]) / self.ki[mask]

        effective_thrust_e = effective_thrust_magnitude / self.vehicle.num_engines
        gimbal_angles_list = self.get_gimbal_commands(control_torque, effective_thrust_e, gimbal_arm)

        if log_flag:
            current_z_unit_vector = rotate_vector_by_quaternion(np.array([0, 0, 1]), current_quaternion)
            desired_z_unit_vector = rotate_vector_by_quaternion(np.array([0, 0, 1]), desired_quaternion)
            attitude_error = desired_z_unit_vector - current_z_unit_vector
            position = state_vector[:3]
            radial_unit_vector = position / np.linalg.norm(position)
            current_dot = np.dot(current_z_unit_vector, radial_unit_vector)
            desired_dot = np.dot(desired_z_unit_vector, radial_unit_vector)
            current_pitch = np.rad2deg(np.pi / 2 - np.arccos(np.clip(current_dot, -1.0, 1.0)))
            desired_pitch = np.rad2deg(np.pi / 2 - np.arccos(np.clip(desired_dot, -1.0, 1.0)))
            logging.info(f"------------------------------------[GUIDANCE]--------------------------------------------")
            logging.info(
                f"current quat: {np.round(current_quaternion, 4)} | current attitude (z_hat): {np.round(current_z_unit_vector, 4)}"
            )
            logging.info(
                f"desired quat: {np.round(desired_quaternion, 4)} | desired attitude (z_hat): {np.round(desired_z_unit_vector, 4)}"
            )
            logging.info(
                f"error quat: {np.round(error_quaternion, 4)} | error attitude (z_hat): {np.round(attitude_error, 4)}"
            )
            logging.info(f"current pitch (deg): {current_pitch:.2f}")
            logging.info(f"desired pitch (deg): {desired_pitch:.2f}")
            logging.info(
                f"pitch error (deg): {(desired_pitch - current_pitch):.2f} | quat error angle (deg): {np.round(np.rad2deg(angle_axis[0]), 4)}"
            )
            logging.info(f"-----------------------------------[CONTROLLER]-------------------------------------------")
            logging.info(f"body frame error (deg): {np.round(np.rad2deg(current_error), 4)}")
            logging.info(
                f"PID p term: {np.round(p_term, 4)} | PID i term: {np.round(i_term, 4)} | PID d term: {np.round(d_term, 4)}"
            )
        return {
            "desired_torque": control_torque,
            "engine_gimbal_angles": gimbal_angles_list,
            "throttle": throttle,
            "propellant_mass": current_propellant_mass,  # Pass to dynamics
        }

    def get_gimbal_commands(self, desired_torque: np.ndarray, effective_thrust_e: float, gimbal_arm: float) -> list:
        """
        Map desired torque to per-engine gimbal angles (pitch, yaw) for all engines.
        Args:
            desired_torque: Desired torque in body frame [Tx, Ty, Tz] (N·m)
            effective_thrust_e: Thrust per engine (N)
            gimbal_arm: Current gimbal arm length (m) from CoM to engine pivot
        Returns:
            List of [pitch, yaw] angles (rad) for each engine
        """
        num_gimbals = self.vehicle.num_engines * 2  # pitch + yaw per engine
        A = np.zeros((5, num_gimbals))  # Rows: Tx, Ty, Tz, netFx, netFy
        for j in range(self.vehicle.num_engines):
            px, py, pz = self.vehicle.engines[j]["position"]
            pz = -gimbal_arm  # Update Z-position with dynamic CoM
            col_pitch = 2 * j
            col_yaw = 2 * j + 1
            # Pitch effects (updated signs for new convention)
            A[0, col_pitch] = pz * effective_thrust_e  # Tx
            A[1, col_pitch] = 0  # Ty
            A[2, col_pitch] = -px * effective_thrust_e  # Tz (roll)
            A[3, col_pitch] = 0  # netFx
            A[4, col_pitch] = -effective_thrust_e  # netFy
            # Yaw effects (updated signs for new convention)
            A[0, col_yaw] = 0  # Tx
            A[1, col_yaw] = pz * effective_thrust_e  # Ty
            A[2, col_yaw] = -py * effective_thrust_e  # Tz (roll)
            A[3, col_yaw] = effective_thrust_e  # netFx
            A[4, col_yaw] = 0  # netFy

        desired_vec = np.concatenate((desired_torque, [0.0, 0.0]))  # Torque + zero lateral force
        gimbal_vec = np.linalg.pinv(A) @ desired_vec  # Minimum-norm solution
        gimbal_vec = np.clip(gimbal_vec, -self.vehicle.engine_gimbal_limit_rad, self.vehicle.engine_gimbal_limit_rad)
        gimbal_angles_list = [
            np.array([gimbal_vec[2 * j], gimbal_vec[2 * j + 1]]) for j in range(self.vehicle.num_engines)
        ]
        return gimbal_angles_list
