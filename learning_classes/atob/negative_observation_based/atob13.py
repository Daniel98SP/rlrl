import calcs

import numpy as np
from gym import spaces

from rocketleaguegym.gamestates.game_state import GameState


class ATOB13:
    def __init__(self, mode='ball'):
        self.mode = mode
        self.point_position = None

    @staticmethod
    def get_spaces():
        observation_space_dictionary = dict()
        observation_space_dictionary['angle_to_point'] = spaces.Box(low=-180.0, high=180.0, shape=(1,), dtype=np.float32)
        observation_space_dictionary['distance_to_point'] = spaces.Box(low=0.0, high=20000, shape=(1,), dtype=np.float32)

        return spaces.Dict(observation_space_dictionary), spaces.Box(-1, 1, shape=(5, ), dtype=np.float32)

    def get_observation(self, state: GameState):
        if self.mode == 'ball':
            self.point_position = state.ball.position
        elif self.mode == 'behind_ball':
            self.point_position = calcs.get_point_behind_ball(state, 1500)
        else:
            self.point_position = self.mode

        observation = dict()
        observation['angle_to_point'] = [calcs.get_yaw_angle_to_point(state, self.point_position)]
        observation['distance_to_point'] = [calcs.get_2d_distance(state.players[0].car_data.position, self.point_position)]

        return observation

    @staticmethod
    def get_reward(previous_obs, actions, current_obs):
        reward = 1000 if current_obs['distance_to_point'][0] <= 200 else 0

        reward += - np.abs(current_obs['angle_to_point'][0]) / 10
        reward += - current_obs['distance_to_point'][0] / 1000

        return reward

    def is_terminal_step(self, state):
        return True if calcs.get_2d_distance(state.players[0].car_data.position, self.point_position) < 200 else False

    @staticmethod
    def format_actions(actions, mode):
        if mode == 'step':
            return {'throttle': np.abs(actions[0]), 'steer': actions[1], 'pitch': 0.0, 'yaw': 0.0, 'roll': 0.0,
                    'jump': -0.005 < actions[2] < 0.005, 'boost': 0 < actions[3], 'handbrake': -0.3 < actions[4] < 0.3}
        elif mode == 'send':
            return [1.0, actions['throttle'], actions['steer'], actions['pitch'], actions['yaw'], actions['roll'], float(actions['jump']), float(actions['boost']), float(actions['handbrake'])]
