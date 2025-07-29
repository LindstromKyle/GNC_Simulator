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
        self, time: float, state_vector: np.ndarray, mission_planner_setpoints: dict
    ) -> np.ndarray:
        position = state_vector[:3]
        velocity = state_vector[3:6]
        radial_unit_vector = position / np.linalg.norm(position)
        velocity_magnitude = np.linalg.norm(velocity)

        mode = mission_planner_setpoints.get("attitude_mode", "prograde")

        if mode == "radial":
            desired_z_vector = radial_unit_vector
        elif mode == "kick":
            kick_direction = mission_planner_setpoints.get("kick_direction")
            kick_angle_rad = mission_planner_setpoints.get("kick_angle_rad")
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
        elif mode == "programmed_pitch":
            # Extract parameters from mission_planner_setpoints
            start_time = mission_planner_setpoints.get("start_time", 0.0)  # Phase start time
            duration = mission_planner_setpoints.get("duration", 100.0)  # Phase duration for interpolation
            initial_pitch_deg = mission_planner_setpoints.get("initial_pitch_deg", 80.0)
            final_pitch_deg = mission_planner_setpoints.get("final_pitch_deg", 45.0)
            kick_direction = mission_planner_setpoints.get("kick_direction", np.array([0.0, 1.0, 0.0]))  # Default east

            # Normalize time progress (0 to 1)
            progress = max(0.0, min(1.0, (time - start_time) / duration))

            # Interpolate pitch angle (from horizontal)
            pitch_deg = initial_pitch_deg + progress * (final_pitch_deg - initial_pitch_deg)
            pitch_rad = np.deg2rad(pitch_deg)

            # Get radial and velocity vectors
            radial_unit_vector = position / np.linalg.norm(position)
            velocity_magnitude = np.linalg.norm(velocity)

            if velocity_magnitude < 1e-3:
                desired_z_vector = radial_unit_vector  # Fallback to radial if low speed

            else:
                # Compute horizontal unit vector in flight plane (perpendicular to radial, towards kick direction)
                horizontal_projection = (
                    kick_direction - np.dot(kick_direction, radial_unit_vector) * radial_unit_vector
                )
                horizontal_unit_vector = horizontal_projection / np.linalg.norm(horizontal_projection)

                # Desired z: sin(pitch) vertical (radial) + cos(pitch) horizontal
                # (For pitch from horizontal: 0° = full horizontal, 90° = full vertical)
                desired_z_vector = np.cos(pitch_rad) * horizontal_unit_vector + np.sin(pitch_rad) * radial_unit_vector
                desired_z_vector /= np.linalg.norm(desired_z_vector)

                # Optional: Blend towards true prograde for late phase (e.g., when progress > 0.5)
                blend_factor = progress  # 0 early (full program), 1 late (full prograde)
                prograde_vector = velocity / velocity_magnitude
                desired_z_vector = (1 - blend_factor) * desired_z_vector + blend_factor * prograde_vector
                desired_z_vector /= np.linalg.norm(desired_z_vector)

        else:
            raise ValueError(f"Unknown attitude mode: {mode}")

        return compute_minimal_quaternion_rotation(desired_z_vector)
