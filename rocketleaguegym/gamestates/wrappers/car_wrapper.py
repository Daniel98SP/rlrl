from rocketleaguegym.gamestates.wrappers.physics_wrapper import PhysicsWrapper
from rocketleaguegym.gamestates.player_data import PlayerData
import numpy as np


class CarWrapper(PhysicsWrapper):

    def __init__(self, team_num: int = -1, id: int = -1, player_data: PlayerData = None):
        if player_data is None:
            super().__init__()
            self.position = np.asarray([id * 100, 0, 17])
            self.rotation: np.ndarray = np.zeros(3)
            self.team_num: int = team_num
            self.id: int = id
            self.boost: float = 0
        else:
            super().__init__(phys_obj=player_data.car_data)
            self._read_from_player_data(player_data)

    def _read_from_player_data(self, player_data: PlayerData):
        self.rotation = player_data.car_data.euler_angles()
        self.team_num = player_data.team_num
        self.id = player_data.car_id
        self.boost = player_data.boost_amount

    def set_rot(self, pitch: float = None, yaw: float = None, roll: float = None):
        if pitch is not None:
            self.rotation[0] = pitch
        if yaw is not None:
            self.rotation[1] = yaw
        if roll is not None:
            self.rotation[2] = roll

    def encode(self) -> list:
        encoded = np.concatenate(((self.id,), self.position, self.linear_velocity, self.angular_velocity, self.rotation, (self.boost,)))

        return encoded.tolist()
