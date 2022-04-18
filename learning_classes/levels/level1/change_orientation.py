import calcs

import numpy as np
from gym import spaces

from rocketleaguegym.communication.message import Message
from rocketleaguegym.gamestates.game_state import GameState

from learning_classes.levels.level0.left_sharp_turn import LeftSharpTurn
from learning_classes.levels.level0.left_turn import LeftTurn
from learning_classes.levels.level0.right_turn import RightTurn
from learning_classes.levels.level0.right_sharp_turn import RightSharpTurn


class ChangeOrientation:
    def __init__(self, goal_orientation_vector):
        self.current_tick = 0
        self.number_of_ticks = 1000
        self.goal_orientation_vector = goal_orientation_vector

    def get_observation(self, state: GameState):
        observation = dict()
        observation['angle'] = [calcs.get_angle_between_vectors(calcs.get_vector_from_rotation(state.players[0].car_data.yaw()), self.goal_orientation_vector)]

        return observation

    @staticmethod
    def get_spaces():
        observation_space = dict()
        observation_space['angle'] = spaces.Box(low=-180.0, high=180.0, shape=(1,), dtype=np.float32)

        return spaces.Dict(observation_space), spaces.Discrete(4)

    def manage_action(self, action, comms):
        model = None
        if action == 0:
            model = LeftSharpTurn()
        elif action == 1:
            model = LeftTurn()
        elif action == 2:
            model = RightTurn()
        elif action == 3:
            model = RightSharpTurn()

        observation = None
        reward = 0
        done = False
        while not model.is_terminal_tick():
            model.execute_action(comms)
            state = GameState(comms.receive_message(header=Message.RLGYM_STATE_MESSAGE_HEADER)[0].body)
            observation = self.get_observation(state)

            reward = 1000 if np.abs(observation['angle'][0]) < 0.2 and state.players[0].car_data.angular_velocity[1] < 0.1 else 0
            reward += - np.sqrt(4 * np.abs(observation['angle'][0]))
            done = done or self.current_tick > self.number_of_ticks or np.abs(observation['angle'][0]) < 0.2 and state.players[0].car_data.angular_velocity[1] < 0.1
            self.current_tick += 1

        return observation, reward, done

