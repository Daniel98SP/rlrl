import numpy as np
from stable_baselines3 import PPO

import calcs
import values
from rocketleaguegym.communication.communication_handler import Message, CommunicationHandler
from rocketleaguegym.gamestates.game_state import GameState
from rocketleaguegym.gamestates.wrappers.state_wrapper import StateWrapper
from learning_classes.atob.negative_observation_based.atob15 import ATOB15
from learning_classes.turn.left_sharp_turn import LeftSharpTurn
from learning_classes.turn.right_sharp_turn import RightSharpTurn
from state_setters.default_state_setter import StateSetter


class TestingEnvironment:
    def __init__(self, game_process):
        self.game_process = game_process
        self.comms = CommunicationHandler()
        self.state = None
        self.learning_class = None
        self.model = None
        self.models = dict()
        self.models['atob15'] = PPO.load('models/atob/atob15')
        self.models['left_sharp_turn'] = PPO.load('models/left_sharp_turn')
        self.models['right_sharp_turn'] = PPO.load('models/right_sharp_turn')

        self.comms.open_pipe(CommunicationHandler.format_pipe_id(0))
        self.comms.send_message(header=Message.RLGYM_CONFIG_MESSAGE_HEADER, body=list({'team_size': 1, 'self_play': 0, 'spawn_opponents': 0, 'tick_skip': 1, 'game_speed': 1}.values()))
        raw_initial_state = StateWrapper(blue_count=1, orange_count=0)
        StateSetter.reset(raw_initial_state)
        self.comms.send_message(header=Message.RLGYM_RESET_GAME_STATE_MESSAGE_HEADER, body=raw_initial_state.format_state())

        self.state = self.state = GameState(self.comms.receive_message(header=Message.RLGYM_STATE_MESSAGE_HEADER)[0].body)

        done = True
        while True:
            if done:
                self.decide_action()
            done = self.step()

    def decide_action(self):
        rival_goal_position = values.ORANGE_GOAL_CENTER if self.state.players[0].team_num == values.BLUE_TEAM else values.BLUE_GOAL_CENTER
        yaw_vector = calcs.get_vector_from_rotation(self.state.players[0].car_data.yaw())
        player_to_ball_vector = calcs.get_2d_vector(self.state.players[0].car_data.position, self.state.ball.position)
        player_to_rival_goal_vector = calcs.get_2d_vector(self.state.players[0].car_data.position, rival_goal_position)

        if np.abs(calcs.get_angle_between_vectors(player_to_ball_vector, player_to_rival_goal_vector)) < 10:
            if np.abs(calcs.get_angle_between_vectors(yaw_vector, player_to_ball_vector)) < 90:
                self.learning_class = ATOB15(mode='ball')
                self.model = self.models['atob3']
            elif calcs.get_angle_between_vectors(yaw_vector, player_to_ball_vector) < -90:
                self.learning_class = RightSharpTurn(calcs.get_2d_vector(self.state.players[0].car_data.position, self.state.ball.position))
                self.model = self.models['right_sharp_turn']
            elif 90 < calcs.get_angle_between_vectors(yaw_vector, player_to_ball_vector):
                self.learning_class = LeftSharpTurn(calcs.get_2d_vector(self.state.players[0].car_data.position, self.state.ball.position))
                self.model = self.models['left_sharp_turn']
        else:
            self.learning_class = ATOB15(mode='behind_ball')
            self.model = self.models['atob3']

        return False

    def step(self):
        observation = self.learning_class.get_observation(self.state)
        self.comms.send_message(header=Message.RLGYM_AGENT_ACTION_IMMEDIATE_RESPONSE_MESSAGE_HEADER, body=self.learning_class.format_actions(self.model.predict(observation)[0], 'send_from_raw'))
        done = self.learning_class.is_terminal_step(self.state)
        self.state = GameState(self.comms.receive_message(header=Message.RLGYM_STATE_MESSAGE_HEADER)[0].body)

        return done

    def close(self):
        self.comms.close_pipe()
        self.game_process.terminate()
