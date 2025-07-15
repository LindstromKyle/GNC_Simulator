"""Vehicle Information"""

import numpy as np

from utils import rotate_vector_by_quaternion


class Vehicle:

    def __init__(self,
                 dry_mass,
                 prop_mass,
                 thrust_magnitude,
                 burn_time,
                 moment_of_inertia,
                 base_drag_coefficient,
                 drag_scaling_coefficient,
                 cross_sectional_area,
                 engine_gimbal_limit,
                 engine_gimbal_arm
                 ):


        self.dry_mass = dry_mass
        self.prop_mass = prop_mass
        self.thrust_magnitude = thrust_magnitude
        self.burn_time = burn_time
        self.moment_of_inertia = moment_of_inertia
        self.base_drag_coefficient = base_drag_coefficient
        self.drag_scaling_coefficient = drag_scaling_coefficient
        self.cross_sectional_area = cross_sectional_area
        self.engine_gimbal_limit = engine_gimbal_limit
        self.engine_gimbal_limit_radians = np.deg2rad(self.engine_gimbal_limit)
        self.engine_gimbal_arm = engine_gimbal_arm

        self.grid_fin_deflections = {
            "Fin 1" : 0.0,
            "Fin 2" : 0.0,
            "Fin 3" : 0.0,
            "Fin 4" : 0.0
        }

    def get_grid_fin_deflections(self, time, state):
        """
        Placeholder for future control logic.
        Return a dict of surface angles (in radians).
        """
        return self.grid_fin_deflections

    def mass(self, time):
        if time < self.burn_time:
            current_propellant_mass = self.prop_mass * (1 - time / self.burn_time)
            return self.dry_mass + current_propellant_mass
        else:
            return self.dry_mass

    def thrust_vector(self, time, quaternion, engine_gimbal_angles):
        if time >= self.burn_time:
            return np.zeros(3), np.zeros(3)

        # Clamp gimbal angles to limits
        gimbal_pitch, gimbal_yaw = np.clip(engine_gimbal_angles, -self.engine_gimbal_limit_radians, self.engine_gimbal_limit_radians)

        # Thrust direction in body frame: nominally [0,0,1], tilted by gimbal
        body_thrust_direction = np.array([
            -np.sin(gimbal_yaw),  # Yaw tilts in body X
            np.sin(gimbal_pitch),  # Pitch tilts in body Y (convention: positive pitch tilts positive Y)
            np.cos(gimbal_pitch) * np.cos(gimbal_yaw)  # Z component
        ])
        body_thrust_direction /= np.linalg.norm(body_thrust_direction)  # Normalize

        # Rotate into inertial frame using current attitude
        inertial_thrust_direction = rotate_vector_by_quaternion(body_thrust_direction, quaternion)

        thrust_vector_force = self.thrust_magnitude * inertial_thrust_direction

        # Torque in body frame: from gimbal offset (perpendicular component)
        # Torque = force_perp * arm; for small angles, ~ thrust * sin(angle) * arm
        thrust_vector_torque = np.array([
            self.thrust_magnitude * np.sin(gimbal_pitch) * self.engine_gimbal_arm,  # Torque around X (from pitch gimbal)
            self.thrust_magnitude * np.sin(gimbal_yaw) * self.engine_gimbal_arm,  # Torque around Y (from yaw gimbal)
            0.0  # No roll from TVC
        ])

        return thrust_vector_force, thrust_vector_torque
