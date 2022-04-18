import calcs

import numpy as np
from gym import spaces

from rocketleaguegym.gamestates.game_state import GameState


class ATOB11:
    def __init__(self, mode='ball'):
        self.mode = mode
        self.point_position = None

    @staticmethod
    def get_spaces():
        observation_space_dictionary = dict()
        observation_space_dictionary['angle_to_ball'] = spaces.Box(low=-180.0, high=180.0, shape=(1,), dtype=np.float32)
        observation_space_dictionary['close_to_ball'] = spaces.Discrete(2)

        return spaces.Dict(observation_space_dictionary), spaces.Box(-1, 1, shape=(5, ), dtype=np.float32)

    def get_observation(self, state: GameState):
        if self.mode == 'ball':
            self.point_position = state.ball.position
        elif self.mode == 'behind_ball':
            self.point_position = calcs.get_point_behind_ball(state, 1500)
        else:
            self.point_position = self.mode

        observation = dict()
        observation['angle_to_ball'] = [calcs.get_velocity_angle_to_point(state.players[0].car_data.position, state.players[0].car_data.linear_velocity, self.point_position)]
        observation['close_to_ball'] = 1 if calcs.get_2d_distance(state.players[0].car_data.position, self.point_position) < 2000 else 0

        return observation

    @staticmethod
    def get_reward(previous_obs, actions, current_obs):
        reward = 10 / np.abs(current_obs['angle_to_ball'][0])

        if -2 < current_obs['angle_to_ball'][0] < 2:
            reward += actions['throttle'] * 10
            reward += 20 if actions['boost'] else 0
            reward += 100 if actions['jump'] and current_obs['close_to_ball'] == 0 else 0

            reward += - np.abs(actions['steer']) * 10
            reward += -20 if actions['handbrake'] else 0
        else:
            reward += -1 if actions['boost'] else 0
            reward += -10 if actions['jump'] else 0

        return reward

    def is_terminal_step(self, state):
        return True if calcs.get_2d_distance(state.players[0].car_data.position, self.point_position) < 200 else False

    @staticmethod
    def format_actions(actions, mode):
        if mode == 'step':
            return {'throttle': actions[0], 'steer': actions[1], 'pitch': 0.0, 'yaw': 0.0, 'roll': 0.0,
                    'jump': -0.005 < actions[2] < 0.005, 'boost': 0 < actions[3], 'handbrake': -0.3 < actions[4] < 0.3}
        elif mode == 'send':
            return [1.0, actions['throttle'], actions['steer'], actions['pitch'], actions['yaw'], actions['roll'], float(actions['jump']), float(actions['boost']), float(actions['handbrake'])]
