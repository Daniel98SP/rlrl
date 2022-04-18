from abc import ABC
from typing import Any

from gym import Env

from rocketleaguegym.communication.communication_handler import Message, CommunicationHandler
from rocketleaguegym.gamestates.game_state import GameState
from rocketleaguegym.gamestates.wrappers.state_wrapper import StateWrapper
from learning_classes.atob.negative_observation_based.atob16 import ATOB16 as LEARNING_CLASS
from state_setters.default_state_setter import StateSetter


class PredictingEnvironment(Env, ABC):
    def __init__(self, game_process):
        super().__init__()

        self.game_process = game_process
        self.comms = CommunicationHandler()
        self.learning_object = None
        self.observation_space, self.action_space = LEARNING_CLASS.get_spaces()

        self.comms.open_pipe(CommunicationHandler.format_pipe_id(0))
        self.comms.send_message(header=Message.RLGYM_CONFIG_MESSAGE_HEADER, body=list({'team_size': 1, 'self_play': 0, 'spawn_opponents': 0, 'tick_skip': 8, 'game_speed': 1000}.values()))

    def reset(self):
        initial_state = StateWrapper(blue_count=1, orange_count=0)
        StateSetter.reset(initial_state, cars_position='random', cars_speed='random', cars_yaw='random', ball_position='random', ball_speed='random')
        self.comms.send_message(header=Message.RLGYM_RESET_GAME_STATE_MESSAGE_HEADER, body=initial_state.format_state())
        self.learning_object = LEARNING_CLASS()
        current_state = GameState(self.comms.receive_message(header=Message.RLGYM_STATE_MESSAGE_HEADER)[0].body)

        return self.learning_object.get_observation(current_state)

    def step(self, raw_actions: Any):
        actions = self.learning_object.format_actions(raw_actions, 'step')
        self.comms.send_message(header=Message.RLGYM_AGENT_ACTION_IMMEDIATE_RESPONSE_MESSAGE_HEADER, body=self.learning_object.format_actions(actions, 'send'))
        current_state = GameState(self.comms.receive_message(header=Message.RLGYM_STATE_MESSAGE_HEADER)[0].body)
        current_observation = self.learning_object.get_observation(current_state)
        done = self.learning_object.is_terminal_step(current_state)

        return current_observation, done

    def close(self):
        self.comms.close_pipe()
        self.game_process.terminate()
