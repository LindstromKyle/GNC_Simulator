import logging
from bdb import effective

import numpy as np
from abc import ABC, abstractmethod

from guidance import Guidance
from mission import MissionPlanner
from state import State  # Import for type hinting
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
        self.ki = ki
        self.kd = kd
        self.guidance = guidance
        self.integral_error = np.zeros(3)  # Accumulator for I term
        self.previous_error = np.zeros(3)
        self.vehicle = vehicle
        self.last_update_time = None

    def update(self, time: float, state_vector: np.ndarray, mission_planner_setpoints: dict, log_flag: bool) -> dict:
        current_quaternion = state_vector[6:10]

        # Throttle
        throttle = mission_planner_setpoints.get("throttle", 1.0)

        # Get desired quaternion from guidance
        desired_quaternion = self.guidance.get_desired_quaternion(time, state_vector, mission_planner_setpoints)
        desired_quaternion /= np.linalg.norm(desired_quaternion)

        # Compute quaterion error (expressed in Body basis vectors) [q_cur^-1(B -> I) then q_des (I -> D)]
        error_quaternion = quaternion_multiply(desired_quaternion, quaternion_inverse(current_quaternion))
        error_quaternion /= np.linalg.norm(error_quaternion)

        # Convert to angle-axis for PID
        angle_axis = quat_to_angle_axis(error_quaternion)
        current_error = angle_axis[0] * angle_axis[1:]  # angle (rad) * Axis

        # TODO: Update this when roll control is introduced
        current_error[2] = 0

        if self.last_update_time:
            dt = time - self.last_update_time
            if dt > 0:
                self.integral_error += current_error * dt
                d_term = self.kd * (current_error - self.previous_error) / dt
            else:
                d_term = np.zeros(3)
        else:
            d_term = np.zeros(3)
        i_term = self.ki * self.integral_error
        self.last_update_time = time
        self.previous_error = current_error
        p_term = self.kp * current_error

        control_torque = p_term + i_term + d_term  # Desired torque

        # Map torque to actuators
        thrust_magnitude = self.vehicle.get_thrust_magnitude(time)
        effective_thrust_magnitude = thrust_magnitude * throttle

        if effective_thrust_magnitude > 0:
            # Compute sin_arg for each axis
            sin_pitch = control_torque[0] / (effective_thrust_magnitude * self.vehicle.engine_gimbal_arm_len)
            sin_yaw = control_torque[1] / (effective_thrust_magnitude * self.vehicle.engine_gimbal_arm_len)
            # Clip to valid arcsin domain (handles |sin_arg| > 1)
            sin_pitch = np.clip(sin_pitch, -1.0, 1.0)
            sin_yaw = np.clip(sin_yaw, -1.0, 1.0)
            # Compute angles
            engine_gimbal_pitch = np.arcsin(sin_pitch)
            engine_gimbal_yaw = np.arcsin(sin_yaw)
            # Ignore roll for now
            engine_gimbal_angles = np.array([engine_gimbal_pitch, engine_gimbal_yaw])
        else:
            engine_gimbal_angles = np.zeros(2)

        if log_flag:
            logging.info(f"------------------------------------[GUIDANCE]--------------------------------------------")
            logging.info(
                f"current quat: {current_quaternion} | attitude: {rotate_vector_by_quaternion(np.array([0,0,1]), current_quaternion)}"
            )
            logging.info(f"desired quat: {desired_quaternion}")
            logging.info(f"error quat: {error_quaternion} | error angle (deg): {np.rad2deg(angle_axis[0])}")
        return {
            "desired_torque": control_torque,
            "engine_gimbal_angles": engine_gimbal_angles,
            "throttle": throttle,
        }  # Expand later
