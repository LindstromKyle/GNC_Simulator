import numpy as np
from abc import ABC, abstractmethod

from state import State
from utils import compute_minimal_quaternion_rotation


class Guidance(ABC):
    """
    Base class for guidance systems. Subclasses compute desired quaternion over time.
    """

    @abstractmethod
    def get_desired_quaternion(
        self, time: float, state_vector: np.ndarray, mission_phase_parameters: dict
    ) -> np.ndarray:
        """
        Compute the desired quaternion based on time and current state.

        Args:
            time (float): Current simulation time (s)
            state_vector (np.ndarray): Current State object
            mission_phase_parameters (dict):

        Returns:
            Desired quaternion [w, x, y, z]
        """
        pass


class ModeBasedGuidance(Guidance):
    def __init__(self):
        pass

    def get_desired_quaternion(
        self, time: float, state_vector: np.ndarray, mission_phase_parameters: dict
    ) -> np.ndarray:
        position = state_vector[:3]
        velocity = state_vector[3:6]
        radial_unit_vector = position / np.linalg.norm(position)
        velocity_magnitude = np.linalg.norm(velocity)

        mode = mission_phase_parameters.get("attitude_mode", "prograde")

        if mode == "radial":
            desired_z_vector = radial_unit_vector
        elif mode == "kick":
            kick_direction = mission_phase_parameters.get("kick_direction")
            kick_angle_rad = mission_phase_parameters.get("kick_angle_rad")
            horizontal_projection = kick_direction - np.dot(kick_direction, radial_unit_vector) * radial_unit_vector
            horizontal_unit_vector = horizontal_projection / np.linalg.norm(horizontal_projection)
            desired_z_vector = (
                np.cos(kick_angle_rad) * radial_unit_vector + np.sin(kick_angle_rad) * horizontal_unit_vector
            )
            desired_z_vector /= np.linalg.norm(desired_z_vector)
        elif mode == "prograde":
            if velocity_magnitude < 1e-3:
                desired_z_vector = radial_unit_vector
            else:
                desired_z_vector = velocity / velocity_magnitude
        elif mode == "retrograde":
            if velocity_magnitude < 1e-3:
                desired_z_vector = -radial_unit_vector
            else:
                desired_z_vector = -velocity / velocity_magnitude
        elif mode == "radial_down":
            desired_z_vector = -radial_unit_vector
        else:
            raise ValueError(f"Unknown attitude mode: {mode}")

        return compute_minimal_quaternion_rotation(desired_z_vector)
