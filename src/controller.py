import numpy as np
from abc import ABC, abstractmethod

from guidance import Guidance
from state import State  # Import for type hinting
from utils import quaternion_multiply, quaternion_inverse, quat_to_angle_axis
from vehicle import Vehicle


class Controller(ABC):
    """
    Base class for controllers. Subclasses implement control logic.
    """

    @abstractmethod
    def update(self, time: float, state: State) -> dict:
        """
        Compute control inputs based on time and current state.

        Args:
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

    def update(self, time: float, state_vector: np.ndarray) -> dict:
        current_quaternion = state_vector[6:10]

        # Get desired quaternion from guidance
        desired_quat = self.guidance.get_desired_quaternion(time, state_vector)
        desired_quat /= np.linalg.norm(desired_quat)

        # Compute quaterion error (expressed in Body basis vectors) [q_cur^-1(B -> I) then q_des (I -> D)]
        error_quaternion = quaternion_multiply(desired_quat, quaternion_inverse(current_quaternion))
        error_quaternion /= np.linalg.norm(error_quaternion)
        # Convert to angle-axis for PID
        angle_axis = quat_to_angle_axis(error_quaternion)
        current_error = angle_axis[0] * angle_axis[1:]  # angle (rad) * Axis

        # PID terms
        p_term = self.kp * current_error
        self.integral_error += current_error  # Simple integral (add dt later if needed)
        i_term = self.ki * self.integral_error
        d_term = self.kd * (current_error - self.previous_error)
        self.previous_error = current_error

        control_torque = p_term + i_term + d_term  # Desired torque

        # Map torque to actuators
        # Example: gimbal = some_mapping(control_torque)  # Implement based on vehicle
        thrust = self.vehicle.thrust_magnitude if time < self.vehicle.burn_time else 0.0
        if thrust > 0:
            engine_gimbal_pitch = np.arcsin(
                control_torque[0] / (thrust * self.vehicle.engine_gimbal_arm)
            )  # For X torque
            engine_gimbal_yaw = np.arcsin(
                control_torque[1] / (thrust * self.vehicle.engine_gimbal_arm)
            )  # For Y torque
            # Ignore roll torque[2] for now (or set to 0)
            engine_gimbal_angles = np.array([engine_gimbal_pitch, engine_gimbal_yaw])
        else:
            engine_gimbal_angles = np.zeros(2)

        return {
            "desired_torque": control_torque,
            "engine_gimbal_angles": engine_gimbal_angles,
        }  # Expand later
