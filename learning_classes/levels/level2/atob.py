import calcs

import numpy as np
from gym import spaces

from rocketleaguegym.communication.message import Message
from rocketleaguegym.gamestates.game_state import GameState

from learning_classes.levels.level1.change_orientation import ChangeOrientation
from learning_classes.levels.level1.straight_to_point import StraightToPoint


class Atob:
    def __init__(self, point_position):
        self.current_tick = 0
        self.number_of_ticks = 10000
        self.point_position = point_position

    def get_observation(self, state: GameState):
        observation = dict()
        observation['angle_to_point'] = [calcs.get_yaw_angle_to_point(state, self.point_position)]

        return observation

    @staticmethod
    def get_spaces():
        observation_space = dict()
        observation_space['angle_to_point'] = spaces.Box(low=-180.0, high=180.0, shape=(1,), dtype=np.float32)

        return spaces.Dict(observation_space), spaces.Discrete(5)

    def manage_action(self, action, comms):
        model = None
        if action == 0:
            model = ChangeOrientation()
        elif action == 1:
            model = StraightToPoint()

        observation = None
        reward = 0
        done = False
        while not model.is_terminal_tick():
            model.execute_action(comms)
            state = GameState(comms.receive_message(header=Message.RLGYM_STATE_MESSAGE_HEADER)[0].body)
            observation = self.get_observation(state)

            reward = 500 if calcs.get_2d_distance(state.players[0].car_data.position, self.point_position) < 300.0 else 0
            reward += - np.sqrt(4 * np.abs(observation['angle_to_point'][0]))
            done = done or self.current_tick > self.number_of_ticks or calcs.get_2d_distance(state.players[0].car_data.position, self.point_position) < 300.0
            self.current_tick += 1

        return observation, reward, done


