import numpy as np
from abc import ABC, abstractmethod
from state import State  # Import for type hinting
from utils import quaternion_multiply, quaternion_inverse, quat_to_axis_angle


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
    def __init__(self, kp: np.ndarray, ki: np.ndarray, kd: np.ndarray, desired_quaternion: np.ndarray, vehicle):
        """
        PID controller for attitude.

        Args:
            kp, ki, kd: Gain arrays (3x1 for roll/pitch/yaw)
            desired_quaternion: Target attitude [w, x, y, z]
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.desired_quat = desired_quaternion / np.linalg.norm(desired_quaternion)  # Normalize
        self.integral_error = np.zeros(3)  # Accumulator for I term
        self.prev_error = np.zeros(3)
        self.vehicle=vehicle

    def update(self, time: float, state: State) -> dict:
        # Compute quaternion error (shortest rotation)
        current_quaternion = state[6:10]
        error_quat = quaternion_multiply(quaternion_inverse(self.desired_quat), current_quaternion)
        # Convert to axis-angle for PID (error vector)
        axis_angle = quat_to_axis_angle(error_quat)
        error = axis_angle[:3] * axis_angle[3]  # Axis * angle (rad)

        # PID terms
        p_term = self.kp * error
        self.integral_error += error  # Simple integral (add dt later if needed)
        i_term = self.ki * self.integral_error
        d_term = self.kd * (error - self.prev_error)
        self.prev_error = error

        control_torque = p_term + i_term + d_term  # Desired torque

        # Map torque to actuators (placeholder: assume TVC for now)
        # Example: gimbal = some_mapping(control_torque)  # Implement based on vehicle
        thrust = self.vehicle.thrust_magnitude if time < self.vehicle.burn_time else 0.0
        if thrust > 0:
            engine_gimbal_pitch = np.arcsin(control_torque[0] / (thrust * self.vehicle.engine_gimbal_arm))  # For X torque
            engine_gimbal_yaw = np.arcsin(control_torque[1] / (thrust * self.vehicle.engine_gimbal_arm))  # For Y torque
            # Ignore roll torque[2] for now (or set to 0)
            engine_gimbal_angles = np.array([engine_gimbal_pitch, engine_gimbal_yaw])
        else:
            engine_gimbal_angles = np.zeros(2)
        return {'desired_torque': control_torque, 'engine_gimbal_angles': engine_gimbal_angles}  # Expand later