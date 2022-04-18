import calcs

import numpy as np
from gym import spaces

from rocketleaguegym.communication.message import Message
from rocketleaguegym.gamestates.game_state import GameState

from learning_classes.levels.level0.front_flip import FrontFlip
from learning_classes.levels.level0.throttle import Throttle


class SupersonicNoBoost:
    def __init__(self):
        self.current_tick = 0
        self.number_of_ticks = 1000

    def get_observation(self, state: GameState):
        observation = dict()
        observation['car_speed'] = [calcs.get_speed(state)]

        return observation

    @staticmethod
    def get_spaces():
        observation_space = dict()
        observation_space['car_speed'] = spaces.Box(low=0.0, high=2300.0, shape=(1,), dtype=np.float32)

        return spaces.Dict(observation_space), spaces.Discrete(2)

    def manage_action(self, action, comms):
        model = None
        if action == 0:
            model = Throttle(repetitions=30)
        elif action == 1:
            model = FrontFlip()

        observation = None
        reward = 0
        done = False
        while not model.is_terminal_tick():
            model.execute_action(comms)
            observation = self.get_observation(GameState(comms.receive_message(header=Message.RLGYM_STATE_MESSAGE_HEADER)[0].body))

            reward = 10000 if observation['car_speed'][0] > 2200.0 else 0
            reward += - np.sqrt(0.5 * (2300.0 - observation['car_speed'][0]))
            done = done or self.current_tick > self.number_of_ticks or observation['car_speed'][0] > 2200.0
            self.current_tick += 1

        return observation, reward, done
