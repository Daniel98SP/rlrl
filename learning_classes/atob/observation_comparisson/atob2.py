import calcs

import numpy as np
from gym import spaces

from rocketleaguegym.gamestates.game_state import GameState


class ATOB2:
    def __init__(self, mode='ball'):
        self.mode = mode
        self.point_position = None

    @staticmethod
    def get_spaces():
        observation_space_dictionary = dict()
        observation_space_dictionary['distance_to_point'] = spaces.Box(low=0, high=50000, shape=(1,), dtype=np.float32)
        observation_space_dictionary['angle_to_point'] = spaces.Box(low=-180, high=180, shape=(1,), dtype=np.float16)

        return spaces.Dict(observation_space_dictionary), spaces.Box(-1, 1, shape=(3, ))

    def get_observation(self, state: GameState):
        if self.mode == 'ball':
            self.point_position = state.ball.position
        elif self.mode == 'behind_ball':
            self.point_position = calcs.get_point_behind_ball(state, 1500)
        else:
            self.point_position = self.mode

        observation = dict()
        observation['distance_to_point'] = [calcs.get_2d_distance(state.players[0].car_data.position, self.point_position)]
        observation['angle_to_point'] = [calcs.get_yaw_angle_to_point(state, self.point_position)]

        return observation

    @staticmethod
    def get_reward(previous_obs, actions, current_obs):
        reward = 1000 if current_obs['distance_to_point'][0] < 200 else 0

        if -2 < current_obs['angle_to_point'][0] < 2:
            reward += 10
            reward += previous_obs['distance_to_point'][0] - current_obs['distance_to_point'][0]

        return reward

    def is_terminal_step(self, state):
        return True if calcs.get_2d_distance(state.players[0].car_data.position, self.point_position) < 200 else False

    @staticmethod
    def format_actions(actions, mode):
        if mode == 'step':
            return {'throttle': actions[0], 'steer': actions[1], 'pitch': 0.0, 'yaw': 0.0, 'roll': 0.0,
                    'jump': 0.0, 'boost': actions[2] > 0, 'handbrake': 0.0}
        elif mode == 'send':
            return [1.0, actions['throttle'], actions['steer'], actions['pitch'], actions['yaw'], actions['roll'], actions['jump'], actions['boost'], actions['handbrake']]
