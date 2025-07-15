import numpy as np
from scipy.spatial.transform import Rotation

class State:
    """
    Position	         Inertial Frame
    Velocity	         Inertial Frame
    Quaternion           Body Rotation in terms of Inertial Axes
    Angular velocity     Body Frame
    """
    def __init__(self, position, velocity, quaternion, angular_velocity):
        self.position = np.array(position)
        self.velocity = np.array(velocity)
        self.quaternion = Rotation.from_quat(quaternion).as_quat()
        self.angular_velocity = np.array(angular_velocity)


    def as_vector(self):
        return np.concatenate([
            self.position,
            self.velocity,
            self.quaternion,
            self.angular_velocity
        ])
