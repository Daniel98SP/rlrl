import numpy as np
from typing import Optional

from rocketleaguegym import calcs


class PhysicsObject(object):
    def __init__(self, position=None, quaternion=None, linear_velocity=None, angular_velocity=None):
        self.position: np.ndarray = position if position is not None else np.zeros(3)

        # ones by default to prevent mathematical errors when converting quat to rot matrix on empty physics state
        self.quaternion: np.ndarray = quaternion if quaternion is not None else np.ones(4)

        self.linear_velocity: np.ndarray = linear_velocity if linear_velocity is not None else np.zeros(3)
        self.angular_velocity: np.ndarray = angular_velocity if angular_velocity is not None else np.zeros(3)
        self._euler_angles: Optional[np.ndarray] = np.zeros(3)
        self._rotation_mtx: Optional[np.ndarray] = np.zeros((3, 3))
        self._has_computed_rot_mtx = False
        self._has_computed_euler_angles = False

    def decode_car_data(self, car_data: np.ndarray):
        self.position = car_data[:3]
        self.quaternion = car_data[3:7]
        self.linear_velocity = car_data[7:10]
        self.angular_velocity = car_data[10:]

    def decode_ball_data(self, ball_data: np.ndarray):
        self.position = ball_data[:3]
        self.linear_velocity = ball_data[3:6]
        self.angular_velocity = ball_data[6:9]

    def forward(self) -> np.ndarray:
        return self.rotation_mtx()[:, 0]

    def right(self) -> np.ndarray:
        return self.rotation_mtx()[:, 1]

    def left(self) -> np.ndarray:
        return self.rotation_mtx()[:, 1] * -1

    def up(self) -> np.ndarray:
        return self.rotation_mtx()[:, 2]

    def pitch(self) -> float:
        return self.euler_angles()[0]

    def yaw(self) -> float:
        return self.euler_angles()[1]

    def roll(self) -> float:
        return self.euler_angles()[2]

    # pitch, yaw, roll
    def euler_angles(self) -> np.ndarray:
        if not self._has_computed_euler_angles:
            self._euler_angles = calcs.quat_to_euler(self.quaternion)
            self._has_computed_euler_angles = True

        return self._euler_angles

    def rotation_mtx(self) -> np.ndarray:
        if not self._has_computed_rot_mtx:
            self._rotation_mtx = calcs.quat_to_rot_mtx(self.quaternion)
            self._has_computed_rot_mtx = True

        return self._rotation_mtx

    def serialize(self):
        repr = []

        if self.position is not None:
            for arg in self.position:
                repr.append(arg)

        if self.quaternion is not None:
            for arg in self.quaternion:
                repr.append(arg)

        if self.linear_velocity is not None:
            for arg in self.linear_velocity:
                repr.append(arg)

        if self.angular_velocity is not None:
            for arg in self.angular_velocity:
                repr.append(arg)

        if self._euler_angles is not None:
            for arg in self._euler_angles:
                repr.append(arg)

        if self._rotation_mtx is not None:
            for arg in self._rotation_mtx.ravel():
                repr.append(arg)

        return repr
