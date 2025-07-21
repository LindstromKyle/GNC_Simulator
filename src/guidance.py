import numpy as np
from abc import ABC, abstractmethod

from state import State


class Guidance(ABC):
    """
    Base class for guidance systems. Subclasses compute desired quaternion over time.
    """

    @abstractmethod
    def get_desired_quaternion(self, time: float, state_vector: np.ndarray) -> np.ndarray:
        """
        Compute the desired quaternion based on time and current state.

        Args:
            time (float): Current simulation time (s)
            state_vector: Current State object

        Returns:
            Desired quaternion [w, x, y, z]
        """
        pass


class GravityTurnGuidance(Guidance):
    def __init__(
        self,
        kick_start_time: float = 20.0,
        kick_angle_deg: float = 2.0,
        prograde_start_time: float = 50.0,
        kick_direction: np.ndarray = np.array([1.0, 0.0, 0.0]),
    ):
        """
        Simple gravity turn guidance: Vertical ascent, then small kick to initiate turn,
        then prograde alignment to follow velocity vector.

        Args:
            kick_start_time: Time (s) to start initial kick (small tilt from vertical).
            kick_angle_deg: Kick angle from vertical (degrees).
            prograde_start_time: Time (s) to switch to prograde tracking.
            kick_direction: Direction for initial kick (e.g., [1,0,0] for +X tilt).
        """
        self.kick_start_time = kick_start_time
        self.kick_angle_deg = kick_angle_deg
        self.kick_angle_rad = np.deg2rad(kick_angle_deg)
        self.prograde_start_time = prograde_start_time
        self.kick_direction = kick_direction / np.linalg.norm(kick_direction)

    def get_desired_quaternion(self, time: float, state_vector: np.ndarray) -> np.ndarray:
        position = state_vector[:3]
        radial_unit_vector = position / np.linalg.norm(position)

        if time < self.kick_start_time:
            # Radial ascent
            desired_z_vector = radial_unit_vector
        elif time < self.prograde_start_time:
            # Kick phase - small tilt from vertical
            # Compute horizontal projection and normalize to unit vector (Gram-Schmidt to ensure orthogonality)
            horizontal_projection = (
                self.kick_direction - np.dot(self.kick_direction, radial_unit_vector) * radial_unit_vector
            )
            horizontal_unit_vector = horizontal_projection / np.linalg.norm(horizontal_projection)
            # Create Z vector with cos component in radial direction and sin component horizontal
            desired_z_vector = (
                np.cos(self.kick_angle_rad) * radial_unit_vector + np.sin(self.kick_angle_rad) * horizontal_unit_vector
            )
            desired_z_vector /= np.linalg.norm(desired_z_vector)
        else:
            # Prograde - align to velocity vector
            velocity_vector = state_vector[3:6]
            velocity_magnitude = np.linalg.norm(velocity_vector)
            if velocity_magnitude < 1e-3:
                # Not moving, align to radial unit vector
                desired_z_vector = radial_unit_vector
            else:
                # Align to velocity uit vector (prograde)
                desired_z_vector = velocity_vector / velocity_magnitude

        # Compute minimal quaternion to rotate body Z [0,0,1] to desired_z
        body_z_vector = np.array([0, 0, 1])
        dot = np.dot(body_z_vector, desired_z_vector)
        dot = np.clip(dot, -1.0, 1.0)

        if dot > 0.99999:
            return np.array([1.0, 0.0, 0.0, 0.0])
        if dot < -0.99999:
            # 180° flip (arbitrary axis)
            return np.array([0.0, 1.0, 0.0, 0.0])

        cross_product = np.cross(body_z_vector, desired_z_vector)
        cross_norm = np.linalg.norm(cross_product)
        if cross_norm < 1e-6:
            axis = np.array([0.0, 0.0, 0.0])
        else:
            axis = cross_product / cross_norm
        angle = np.arccos(dot)
        sin_half = np.sin(angle / 2.0)
        cos_half = np.cos(angle / 2.0)
        quat = np.array([cos_half, sin_half * axis[0], sin_half * axis[1], sin_half * axis[2]])
        return quat / np.linalg.norm(quat)


class ReturnGuidance(Guidance):
    def get_desired_quaternion(self, time: float, state_vector: np.ndarray) -> np.ndarray:
        position = state_vector[:3]
        velocity = state_vector[3:6]

        velocity_magnitude = np.linalg.norm(velocity)
        if velocity_magnitude < 1e-3:
            # Fall back to radial
            desired_z_vector = -position / np.linalg.norm(position)  # Point down for landing
        else:
            # Retrograde: Point Z opposite to velocity (engine towards velocity for retro burn)
            desired_z_vector = -velocity / velocity_magnitude

        # Compute minimal quaternion to rotate body Z [0,0,1] to desired_z
        body_z_vector = np.array([0, 0, 1])
        dot = np.dot(body_z_vector, desired_z_vector)
        dot = np.clip(dot, -1.0, 1.0)

        if dot > 0.99999:
            return np.array([1.0, 0.0, 0.0, 0.0])
        if dot < -0.99999:
            # 180° flip (arbitrary axis)
            return np.array([0.0, 1.0, 0.0, 0.0])

        cross_product = np.cross(body_z_vector, desired_z_vector)
        cross_norm = np.linalg.norm(cross_product)
        if cross_norm < 1e-6:
            axis = np.array([0.0, 0.0, 0.0])
        else:
            axis = cross_product / cross_norm
        angle = np.arccos(dot)
        sin_half = np.sin(angle / 2.0)
        cos_half = np.cos(angle / 2.0)
        quat = np.array([cos_half, sin_half * axis[0], sin_half * axis[1], sin_half * axis[2]])
        return quat / np.linalg.norm(quat)


class OrbitalInsertionGuidance(Guidance):
    def __init__(
        self,
        target_apoapsis: float = 200000.0,  # Target apoapsis altitude (m); optional for shutdown logic
        circularize: bool = False,  # Future: Add coast + circular burn if True
    ):
        self.target_apoapsis = target_apoapsis
        self.circularize = circularize  # Placeholder for advanced features

    def get_desired_quaternion(self, time: float, state_vector: np.ndarray) -> np.ndarray:
        position = state_vector[:3]
        velocity = state_vector[3:6]
        velocity_magnitude = np.linalg.norm(velocity)

        if velocity_magnitude < 1e-3:
            # Fallback: Align to radial if not moving (unlikely post-separation)
            desired_z_vector = position / np.linalg.norm(position)
        else:
            # Immediately align to prograde (velocity direction) to continue gravity turn
            desired_z_vector = velocity / velocity_magnitude

        # Optional: Future logic for shutdown or circularization
        # e.g., Compute current apoapsis; if > target and circularize, align horizontal or something
        # For now, just prograde

        # Compute quaternion
        body_z_vector = np.array([0, 0, 1])
        dot = np.dot(body_z_vector, desired_z_vector)
        dot = np.clip(dot, -1.0, 1.0)

        if dot > 0.99999:
            return np.array([1.0, 0.0, 0.0, 0.0])
        if dot < -0.99999:
            return np.array([0.0, 1.0, 0.0, 0.0])

        cross_product = np.cross(body_z_vector, desired_z_vector)
        cross_norm = np.linalg.norm(cross_product)
        if cross_norm < 1e-6:
            axis = np.array([1.0, 0.0, 0.0])  # Arbitrary axis for zero cross
        else:
            axis = cross_product / cross_norm
        angle = np.arccos(dot)
        sin_half = np.sin(angle / 2.0)
        cos_half = np.cos(angle / 2.0)
        quat = np.array([cos_half, sin_half * axis[0], sin_half * axis[1], sin_half * axis[2]])
        return quat / np.linalg.norm(quat)
