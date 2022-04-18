import random
from abc import ABC
from typing import Any

import numpy as np
from gym import Env

from rocketleaguegym.communication.communication_handler import Message, CommunicationHandler
from rocketleaguegym.gamestates.game_state import GameState
from rocketleaguegym.gamestates.wrappers.state_wrapper import StateWrapper
from learning_classes.levels.level1.straight_to_point import StraightToPoint as LEARNING_CLASS
from state_setters.default_state_setter import StateSetter


class TrainingEnvironment(Env, ABC):
    def __init__(self, game_process):
        super().__init__()

        self.game_process = game_process
        self.comms = CommunicationHandler()
        self.learning_class = None
        self.observation_space, self.action_space = LEARNING_CLASS.get_spaces()

        self.comms.open_pipe(CommunicationHandler.format_pipe_id(0))
        self.comms.send_message(header=Message.RLGYM_CONFIG_MESSAGE_HEADER, body=list({'team_size': 1, 'self_play': 0, 'spawn_opponents': 0, 'tick_skip': 1, 'game_speed': 100}.values()))

    def reset(self):
        initial_state = StateWrapper(blue_count=1, orange_count=0)
        StateSetter.reset(initial_state, cars_position=[0, -5000, 17], cars_speed='random', cars_yaw=np.pi/2, ball_position='outside', ball_speed='zero')
        self.comms.send_message(header=Message.RLGYM_RESET_GAME_STATE_MESSAGE_HEADER, body=initial_state.format_state())
        self.learning_class = LEARNING_CLASS([0, random.uniform(-4000, 5000), 17])

        return self.learning_class.get_observation(GameState(self.comms.receive_message(header=Message.RLGYM_STATE_MESSAGE_HEADER)[0].body))

    def step(self, action: Any):
        observation, reward, done = self.learning_class.manage_action(action, self.comms)

        return observation, reward, done, {}

    def close(self):
        self.comms.close_pipe()
        self.game_process.terminate()
