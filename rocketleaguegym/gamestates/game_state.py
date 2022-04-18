import numpy as np
from typing import List, Optional
from rocketleaguegym.gamestates.player_data import PlayerData
from rocketleaguegym.gamestates.physics_object import PhysicsObject


class GameState(object):
    _BOOST_PADS_INFO_LENGTH = 34
    _BALL_STATE_INFO_LENGTH = 18
    _PLAYER_INFO_LENGTH = 38
    _PLAYER_CAR_STATE_INFO_LENGTH = 13
    _PLAYER_TERTIARY_INFO_LENGTH = 10

    def __init__(self, raw_state: List[float] = None):
        self.game_type: int = 0
        self.blue_score: int = -1
        self.orange_score: int = -1
        self.last_touch: Optional[int] = -1
        self.players: List[PlayerData] = []
        self.ball: PhysicsObject = PhysicsObject()
        self.boost_pads: np.ndarray = np.zeros(GameState._BOOST_PADS_INFO_LENGTH, dtype=np.float32)

        if raw_state is not None:
            self._decode(raw_state)

    def _decode(self, raw_state: List[float]):
        assert type(raw_state) == list, "UNABLE TO DECODE STATE OF TYPE {}".format(type(raw_state))

        start = 3

        num_ball_packets = 1
        num_player_packets = int((len(raw_state) - num_ball_packets * GameState._BALL_STATE_INFO_LENGTH - start - GameState._BOOST_PADS_INFO_LENGTH) / GameState._PLAYER_INFO_LENGTH)

        # ticks = int(raw_state[0])
        self.blue_score = int(raw_state[1])
        self.orange_score = int(raw_state[2])

        self.boost_pads[:] = raw_state[start:start + GameState._BOOST_PADS_INFO_LENGTH]
        start += GameState._BOOST_PADS_INFO_LENGTH

        ball_data = raw_state[start:start + GameState._BALL_STATE_INFO_LENGTH]
        self.ball.decode_ball_data(np.asarray(ball_data))
        start += GameState._BALL_STATE_INFO_LENGTH

        for i in range(num_player_packets):
            player = self._decode_player(raw_state[start:start + GameState._PLAYER_INFO_LENGTH])
            self.players.append(player)
            start += GameState._PLAYER_INFO_LENGTH

            if player.ball_touched:
                self.last_touch = player.car_id
                
        self.players = sorted(self.players, key=lambda p: p.car_id)

    @staticmethod
    def _decode_player(full_player_data):
        player_data = PlayerData()

        start = 2

        car_data = full_player_data[start:start + GameState._PLAYER_CAR_STATE_INFO_LENGTH]
        player_data.car_data.decode_car_data(np.asarray(car_data))
        start += GameState._PLAYER_CAR_STATE_INFO_LENGTH * 2

        tertiary_data = full_player_data[start:start + GameState._PLAYER_TERTIARY_INFO_LENGTH]

        player_data.match_goals = int(tertiary_data[0])
        player_data.match_saves = int(tertiary_data[1])
        player_data.match_shots = int(tertiary_data[2])
        player_data.match_demolishes = int(tertiary_data[3])
        player_data.boost_pickups = int(tertiary_data[4])
        player_data.is_demoed = True if tertiary_data[5] > 0 else False
        player_data.on_ground = True if tertiary_data[6] > 0 else False
        player_data.ball_touched = True if tertiary_data[7] > 0 else False
        player_data.has_flip = True if tertiary_data[8] > 0 else False
        player_data.boost_amount = float(tertiary_data[9])
        player_data.car_id = int(full_player_data[0])
        player_data.team_num = int(full_player_data[1])

        return player_data
