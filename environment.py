

import numpy as np


class Environment:
    def __init__(self):
        self.earth_radius = 6371000
        self.earth_mass = 5.972e24
        self.gravitational_constant = 6.67430e-11

    def gravitational_force(self, position, vehicle_mass):
        radius = np.linalg.norm(position)

        # Protect against divide by zero
        if radius < 1e-3:
            return np.zeros(3)

        return (-1 * self.gravitational_constant * self.earth_mass * vehicle_mass * position) / (radius ** 3)

    def atmospheric_density(self, altitude):
        sea_level_density = 1.225
        scale_height = 8500
        return sea_level_density * np.exp(-1 * altitude / scale_height)

    def drag_force(self, position, velocity, vehicle):
        altitude = np.linalg.norm(position) - self.earth_radius

        # Protect against divide by zero
        if altitude < 0:
            altitude = 0

        density = self.atmospheric_density(altitude)
        velocity_magnitude = np.linalg.norm(velocity)

        # Protect against divide by zero
        if velocity_magnitude < 1e-3:
            return np.zeros(3)

        drag_unit_vector = -velocity / velocity_magnitude
        drag_magnitude = 0.5 * density * (velocity_magnitude ** 2) * vehicle.drag_coefficient * vehicle.cross_sectional_area
        return drag_magnitude * drag_unit_vector