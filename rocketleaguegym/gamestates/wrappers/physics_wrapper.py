from rocketleaguegym.gamestates.physics_object import PhysicsObject
import numpy as np


class PhysicsWrapper(object):

    def __init__(self, phys_obj: PhysicsObject = None):
        if phys_obj is None:
            self.position: np.ndarray = np.asarray([0, 0, 93])
            self.linear_velocity: np.ndarray = np.zeros(3)
            self.angular_velocity: np.ndarray = np.zeros(3)
        else:
            self._read_from_physics_object(phys_obj)

    def _read_from_physics_object(self, phys_obj: PhysicsObject):
        self.position = phys_obj.position
        self.linear_velocity = phys_obj.linear_velocity
        self.angular_velocity = phys_obj.angular_velocity

    def set_position(self, x: float = None, y: float = None, z: float = None):
        if x is not None:
            self.position[0] = x
        if y is not None:
            self.position[1] = y
        if z is not None:
            self.position[2] = z

    def set_linear_velocity(self, x: float = None, y: float = None, z: float = None):
        if x is not None:
            self.linear_velocity[0] = x
        if y is not None:
            self.linear_velocity[1] = y
        if z is not None:
            self.linear_velocity[2] = z

    def set_angular_velocity(self, x: float = None, y: float = None, z: float = None):
        if x is not None:
            self.angular_velocity[0] = x
        if y is not None:
            self.angular_velocity[1] = y
        if z is not None:
            self.angular_velocity[2] = z

    def encode(self) -> list:
        encoded = np.concatenate((self.position, self.linear_velocity, self.angular_velocity))
        return encoded.tolist()
