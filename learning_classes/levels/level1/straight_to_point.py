import calcs

import numpy as np
from gym import spaces

from rocketleaguegym.communication.message import Message
from rocketleaguegym.gamestates.game_state import GameState

from learning_classes.levels.level0.throttle import Throttle
from learning_classes.levels.level0.front_flip import FrontFlip
from learning_classes.levels.level0.boost import Boost


class StraightToPoint:
    def __init__(self, point_coordinates):
        self.current_tick = 0
        self.number_of_ticks = 10000
        self.point_coordinates = point_coordinates

    def get_observation(self, state: GameState):
        observation = dict()
        observation['distance'] = [calcs.get_2d_distance(state.players[0].car_data.position, self.point_coordinates)]
        observation['car_speed'] = [calcs.get_speed(state)]
        observation['boost_amount'] = [state.players[0].boost_amount]

        return observation

    @staticmethod
    def get_spaces():
        observation_space = dict()
        observation_space['distance'] = spaces.Box(low=0.0, high=20000.0, shape=(1,), dtype=np.float32)
        observation_space['car_speed'] = spaces.Box(low=0.0, high=2300.0, shape=(1,), dtype=np.float32)
        observation_space['boost_amount'] = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

        return spaces.Dict(observation_space), spaces.Discrete(3)

    def manage_action(self, action, comms):
        model = None
        if action == 0:
            model = Throttle(repetitions=30)
        elif action == 1:
            model = Boost(repetitions=30)
        elif action == 2:
            model = FrontFlip()

        observation = None
        reward = 0
        done = False
        while not model.is_terminal_tick():
            model.execute_action(comms)
            observation = self.get_observation(GameState(comms.receive_message(header=Message.RLGYM_STATE_MESSAGE_HEADER)[0].body))

            reward = 10000 if observation['distance'][0] < 300.0 else 0
            done = done or self.current_tick > self.number_of_ticks or observation['distance'][0] < 300.0
            self.current_tick += 1

        return observation, reward, done


