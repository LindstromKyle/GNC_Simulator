"""Vehicle Information"""

import numpy as np

from utils import rotate_vector_by_quaternion


class Vehicle:

    def __init__(
        self,
        dry_mass: float,
        initial_prop_mass: float,
        base_thrust_magnitude: float,
        average_isp: float,
        moment_of_inertia: np.ndarray,
        base_drag_coefficient: float,
        drag_scaling_coefficient: float,
        cross_sectional_area: float,
        engine_gimbal_limit_deg: float,
        engine_gimbal_arm_len: float,
    ):

        self.dry_mass = dry_mass
        self.initial_propellant_mass = initial_prop_mass
        self.base_thrust_magnitude = base_thrust_magnitude
        self.average_isp = average_isp
        self.g0 = 9.80665
        self.mdot_max = base_thrust_magnitude / (average_isp * self.g0) if average_isp > 0 else 0.0
        self.moment_of_inertia = moment_of_inertia
        self.base_drag_coefficient = base_drag_coefficient
        self.drag_scaling_coefficient = drag_scaling_coefficient
        self.cross_sectional_area = cross_sectional_area
        self.engine_gimbal_limit_deg = engine_gimbal_limit_deg
        self.engine_gimbal_limit_rad = np.deg2rad(self.engine_gimbal_limit_deg)
        self.engine_gimbal_arm_len = engine_gimbal_arm_len

        self.grid_fin_deflections = {
            "Fin 1": 0.0,
            "Fin 2": 0.0,
            "Fin 3": 0.0,
            "Fin 4": 0.0,
        }

    def get_grid_fin_deflections(self, time, state):
        """
        Placeholder for future control logic.
        Return a dict of surface angles (in radians).
        """
        return self.grid_fin_deflections

    def thrust_vector(self, time, quaternion, engine_gimbal_angles, throttle: float = 1.0):
        """

        Args:
            time ():
            quaternion ():
            engine_gimbal_angles ():
            throttle (float):

        Returns:

        """
        # If throttle is zero or invalid, thrust is zero
        if throttle <= 0:
            return np.zeros(3), np.zeros(3)

        # Clamp gimbal angles to limits
        gimbal_pitch, gimbal_yaw = np.clip(
            engine_gimbal_angles,
            -self.engine_gimbal_limit_rad,
            self.engine_gimbal_limit_rad,
        )

        # Thrust direction in body frame: nominally [0,0,1], tilted by gimbal
        body_thrust_direction = np.array(
            [
                -np.sin(gimbal_yaw),  # Yaw tilts in body X
                np.sin(gimbal_pitch),  # Pitch tilts in body Y (convention: positive pitch tilts positive Y)
                np.cos(gimbal_pitch) * np.cos(gimbal_yaw),  # Z component
            ]
        )
        body_thrust_direction /= np.linalg.norm(body_thrust_direction)  # Normalize

        # Rotate into inertial frame using current attitude
        inertial_thrust_direction = rotate_vector_by_quaternion(body_thrust_direction, quaternion)

        effective_thrust_magnitude = self.base_thrust_magnitude * throttle

        thrust_vector_force = effective_thrust_magnitude * inertial_thrust_direction

        # Torque in body frame: from gimbal offset (perpendicular component)
        # Torque = force_perp * arm; for small angles, ~ thrust * sin(angle) * arm
        thrust_vector_torque = np.array(
            [
                effective_thrust_magnitude
                * np.sin(gimbal_pitch)
                * self.engine_gimbal_arm_len,  # Torque around X (from pitch gimbal)
                effective_thrust_magnitude
                * np.sin(gimbal_yaw)
                * self.engine_gimbal_arm_len,  # Torque around Y (from yaw gimbal)
                0.0,  # No roll from TVC
            ]
        )

        return thrust_vector_force, thrust_vector_torque

    # Optional: If you need thrust magnitude separately (e.g., for logging)
    def get_thrust_magnitude(self, throttle: float = 1.0):
        return self.base_thrust_magnitude * max(min(throttle, 1.0), 0.0)
