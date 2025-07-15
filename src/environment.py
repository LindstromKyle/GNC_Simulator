import numpy as np

from utils import rotate_vector_by_quaternion


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

    def drag_force(self, position, velocity, vehicle, quaternion):

        altitude = np.linalg.norm(position) - self.earth_radius

        # Protect against divide by zero
        if altitude < 0:
            altitude = 0

        density = self.atmospheric_density(altitude)
        velocity_magnitude = np.linalg.norm(velocity)

        # Protect against divide by zero
        if velocity_magnitude < 1e-3:
            return np.zeros(3)

        # Direction of drag force (opposite of velocity)
        drag_unit_vector = -velocity / velocity_magnitude

        # Transform body frame Z axis to inertial frame
        body_frame_z_axis = np.array([0, 0, 1])
        inertial_frame_z_axis = rotate_vector_by_quaternion(body_frame_z_axis, quaternion)

        # Compute angle of attack (radians)
        cos_alpha = np.dot(velocity, inertial_frame_z_axis) / velocity_magnitude
        # Clamp for numerical safety
        cos_alpha = np.clip(cos_alpha, -1.0, 1.0)
        angle_of_attack = np.arccos(cos_alpha)

        # Adjust drag coefficient based on AoA
        total_drag_coefficient = vehicle.base_drag_coefficient + vehicle.drag_scaling_coefficient * np.sin(angle_of_attack) ** 2

        # Compute drag magnitude
        drag_magnitude = 0.5 * density * velocity_magnitude ** 2 * total_drag_coefficient * vehicle.cross_sectional_area

        return drag_magnitude * drag_unit_vector

    def aerodynamic_torque(self, position, velocity, quaternion, angular_velocity, vehicle):
        # Get control surface deflections
        deflections = vehicle.get_grid_fin_deflections(time=None, state=None)

        # Eventually add angle of attack math here for cross-sectional area
        # Placeholder: return zero for now
        return np.zeros(3)