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
                 drag_coefficient,
                 cross_sectional_area,

                 ):


        self.dry_mass = dry_mass
        self.prop_mass = prop_mass
        self.thrust_magnitude = thrust_magnitude
        self.burn_time = burn_time
        self.moment_of_inertia = moment_of_inertia
        self.drag_coefficient = drag_coefficient
        self.cross_sectional_area = cross_sectional_area

    def mass(self, time):
        if time < self.burn_time:
            current_propellant_mass = self.prop_mass * (1 - time / self.burn_time)
            return self.dry_mass + current_propellant_mass
        else:
            return self.dry_mass

    def thrust_vector(self, time, quaternion):
        if time >= self.burn_time:
            return np.zeros(3)

        # Thrust in body frame will be along Z axis
        body_thrust_direction = np.array([0, 0, 1])

        # Rotate into inertial frame using current attitude
        inertial_thrust_direction = rotate_vector_by_quaternion(body_thrust_direction, quaternion)

        return self.thrust_magnitude * inertial_thrust_direction
