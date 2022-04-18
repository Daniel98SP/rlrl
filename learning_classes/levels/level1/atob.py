import calcs

import numpy as np
from gym import spaces

from rocketleaguegym.communication.message import Message
from rocketleaguegym.gamestates.game_state import GameState

from learning_classes.levels.level0.boost import Boost
from learning_classes.levels.level0.front_flip import FrontFlip
from learning_classes.levels.level0.left_sharp_turn import LeftSharpTurn
from learning_classes.levels.level0.left_turn import LeftTurn
from learning_classes.levels.level0.right_turn import RightTurn
from learning_classes.levels.level0.right_sharp_turn import RightSharpTurn


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

        return spaces.Dict(observation_space), spaces.Discrete(6)

    def manage_action(self, action, comms):
        model = None
        if action == 0:
            model = Boost(10)
        elif action == 1:
            model = FrontFlip()
        elif action == 2:
            model = LeftSharpTurn(1)
        elif action == 3:
            model = LeftTurn(1)
        elif action == 4:
            model = RightTurn(1)
        elif action == 5:
            model = RightSharpTurn(1)

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


