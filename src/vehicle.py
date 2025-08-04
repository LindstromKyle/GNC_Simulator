"""Vehicle Information"""

from abc import abstractmethod, ABC

import numpy as np

from utils import rotate_vector_by_quaternion


class Vehicle(ABC):
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
        dry_com_z: float,  # New: Z-position of dry mass CoM above base (m)
        prop_com_z: float,  # New: Z-position of propellant CoM above base (m)
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
        self.engine_lever_arm = engine_gimbal_arm_len  # Nominal arm length
        self.dry_com_z = dry_com_z
        self.prop_com_z = prop_com_z

        # Abstract properties/methods for stage-specific engine config
        self._setup_engines()

        # TODO: This needs to live somewhere else
        self.grid_fin_deflections = {
            "Fin 1": 0.0,
            "Fin 2": 0.0,
            "Fin 3": 0.0,
            "Fin 4": 0.0,
        }

    # TODO: This needs to live somewhere else like above
    def get_grid_fin_deflections(self, time, state):
        """
        Placeholder for future control logic.
        Return a dict of surface angles (in radians).
        """
        return self.grid_fin_deflections

    @abstractmethod
    def _setup_engines(self):
        """Stage-specific engine configuration (positions, count, etc.)."""
        pass

    def get_gimbal_arm(self, propellant_mass: float) -> float:
        total_mass = self.dry_mass + propellant_mass
        if total_mass <= 0:
            return self.dry_com_z
        com_z = (self.dry_mass * self.dry_com_z + propellant_mass * self.prop_com_z) / total_mass
        return com_z

    def thrust_vector(self, time, quaternion, gimbal_angles_list, throttle: float = 1.0, propellant_mass: float = 0.0):
        """
        Compute total thrust force and torque from all engines.
        Args:
            time: Simulation time (s)
            quaternion: Current attitude quaternion [w, x, y, z]
            gimbal_angles_list: List of [pitch, yaw] angles (rad) for each engine
            throttle: Throttle level (0 to 1)
            propellant_mass: Current propellant mass (kg) for dynamic CoM
        Returns:
            thrust_vector_force: Total thrust force in inertial frame (N)
            thrust_vector_torque: Total torque in body frame (N·m)
        """
        if throttle <= 0 or len(gimbal_angles_list) != self.num_engines:
            return np.zeros(3), np.zeros(3)

        # Get dynamic gimbal arm based on propellant mass
        gimbal_arm = self.get_gimbal_arm(propellant_mass)

        thrust_force = np.zeros(3)  # Inertial frame
        thrust_vector_torque = np.zeros(3)  # Body frame
        for i in range(self.num_engines):
            gimbal_pitch, gimbal_yaw = np.clip(
                gimbal_angles_list[i],
                -self.engine_gimbal_limit_rad,
                self.engine_gimbal_limit_rad,
            )
            # Thrust direction in body frame
            body_thrust_direction = np.array(
                [np.sin(gimbal_yaw), -np.sin(gimbal_pitch), np.cos(gimbal_pitch) * np.cos(gimbal_yaw)]
            )
            body_thrust_direction /= np.linalg.norm(body_thrust_direction)
            engine_thrust = self.thrust_per_engine * throttle
            body_force = engine_thrust * body_thrust_direction
            inertial_force = rotate_vector_by_quaternion(body_force, quaternion)
            thrust_force += inertial_force
            # Torque: use dynamic gimbal arm for pitch/yaw, engine position for roll
            engine_pos = self.engines[i]["position"].copy()
            engine_pos[2] = -gimbal_arm  # Update Z-position with dynamic CoM
            body_torque = np.cross(engine_pos, body_force)
            thrust_vector_torque += body_torque

        return thrust_force, thrust_vector_torque

    def get_thrust_magnitude(self, time, throttle: float = 1.0):
        """
        Compute total thrust magnitude.
        """
        return self.base_thrust_magnitude * max(min(throttle, 1.0), 0.0)


class Falcon9FirstStage(Vehicle):
    def _setup_engines(self):
        self.num_engines = 9
        self.thrust_per_engine = self.base_thrust_magnitude / self.num_engines
        self.mdot_per_engine = self.mdot_max / self.num_engines
        self.engine_radius = 1.5  # Outer engine radius (m)
        pz = -self.engine_lever_arm
        angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        outer_positions = [
            np.array([self.engine_radius * np.cos(theta), self.engine_radius * np.sin(theta), pz]) for theta in angles
        ]
        center_position = np.array([0.0, 0.0, pz])
        self.engines = [{"position": pos} for pos in outer_positions + [center_position]]


class Falcon9SecondStage(Vehicle):
    def _setup_engines(self):
        self.num_engines = 1  # Single Merlin Vacuum engine
        self.thrust_per_engine = self.base_thrust_magnitude
        self.mdot_per_engine = self.mdot_max
        self.engine_radius = 0.0  # Centered
        pz = -self.engine_lever_arm
        center_position = np.array([0.0, 0.0, pz])
        self.engines = [{"position": center_position}]
