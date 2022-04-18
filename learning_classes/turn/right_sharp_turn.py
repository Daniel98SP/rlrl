import calcs

import numpy as np
from gym import spaces

from rocketleaguegym.gamestates.game_state import GameState


class RightSharpTurn:
    def __init__(self, goal_yaw_vector):
        self.goal_yaw_vector = goal_yaw_vector

    @staticmethod
    def get_spaces():
        observation_space_dictionary = dict()
        observation_space_dictionary['angle'] = spaces.Box(low=-180.0, high=180.0, shape=(1,), dtype=np.float32)

        return spaces.Dict(observation_space_dictionary), spaces.Box(low=np.array([-1.0, -1.0, -1.0]), high=np.array([1.0, 1.0, 1.0]), dtype=np.float32)

    def get_observation(self, state: GameState):
        player_yaw_vector = calcs.get_vector_from_rotation(state.players[0].car_data.yaw())

        observation = dict()
        observation['angle'] = [calcs.get_angle_between_vectors(self.goal_yaw_vector, player_yaw_vector)]

        return observation

    def get_reward(self, previous_obs, actions, current_obs):
        reward = 4000 if np.abs(current_obs['angle'][0]) < 1 else 0
        reward += - np.sqrt(4 * np.abs(current_obs['angle'][0]))

        return reward

    def is_terminal_step(self, current_obs):
        return True if np.abs(current_obs['angle'][0]) < 1 else False

    def format_actions(self, actions, mode):
        if mode == 'step':
            return {'throttle': np.abs(actions[0]), 'steer': np.abs(actions[1]), 'pitch': 0.0, 'yaw': 0.0, 'roll': 0.0,
                    'jump': 0.0, 'boost': 0.0, 'handbrake': actions[2]}
        elif mode == 'send':
            return [1.0, actions['throttle'], actions['steer'], actions['pitch'], actions['yaw'], actions['roll'], float(actions['jump']), float(actions['boost']), float(actions['handbrake'])]
        elif mode == 'send_from_raw':
            actions = self.format_actions(actions, 'step')
            actions = self.format_actions(actions, 'send')
            return actions
