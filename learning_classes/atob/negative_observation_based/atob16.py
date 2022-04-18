import calcs

import numpy as np
from gym import spaces

from rocketleaguegym.gamestates.game_state import GameState


class ATOB16:
    def __init__(self, mode='ball'):
        self.mode = mode
        self.point_position = None

    @staticmethod
    def get_spaces():
        observation_space_dictionary = dict()
        observation_space_dictionary['angle_to_point'] = spaces.Box(low=-180.0, high=180.0, shape=(1,), dtype=np.float32)
        observation_space_dictionary['distance_to_point'] = spaces.Box(low=0.0, high=20000.0, shape=(1,), dtype=np.float32)
        observation_space_dictionary['point_relative_velocity'] = spaces.Box(low=-6000.0, high=6000.0, shape=(1,), dtype=np.float32)

        return spaces.Dict(observation_space_dictionary), spaces.Box(-1, 1, shape=(4, ), dtype=np.float32)

    def get_observation(self, state: GameState):
        if self.mode == 'ball':
            self.point_position = state.ball.position
            car_to_ball_vector = calcs.get_2d_vector(state.players[0].car_data.position, state.ball.position)
            car_to_ball_unit_vector = car_to_ball_vector / np.linalg.norm(car_to_ball_vector)
            new_base = np.array([
                [car_to_ball_unit_vector[0], car_to_ball_unit_vector[1], 0],
                [car_to_ball_unit_vector[1], -car_to_ball_unit_vector[0], 0],
                [0, 0, 1]
            ])

            relative_speed = np.linalg.inv(new_base).dot(state.ball.linear_velocity)
        elif self.mode == 'behind_ball':
            self.point_position = calcs.get_point_behind_ball(state, 1500)
            self.point_position = state.ball.position
            car_to_ball_vector = calcs.get_2d_vector(state.players[0].car_data.position, state.ball.position)
            car_to_ball_unit_vector = car_to_ball_vector / np.linalg.norm(car_to_ball_vector)
            new_base = np.array([
                [car_to_ball_unit_vector[0], car_to_ball_unit_vector[1], 0],
                [car_to_ball_unit_vector[1], -car_to_ball_unit_vector[0], 0],
                [0, 0, 1]
            ])

            relative_speed = np.linalg.inv(new_base).dot(state.ball.linear_velocity)
        else:
            self.point_position = self.mode
            relative_speed = np.array([0, 0, 0])

        observation = dict()
        observation['angle_to_point'] = [calcs.get_yaw_angle_to_point(state, self.point_position)]
        observation['distance_to_point'] = [calcs.get_2d_distance(state.players[0].car_data.position, self.point_position)]
        observation['point_relative_velocity'] = [relative_speed[1]]

        return observation

    @staticmethod
    def get_reward(previous_obs, actions, current_obs):
        reward = 1000 if current_obs['distance_to_point'][0] < 200 else 0
        reward += - np.sqrt(4 * np.abs(current_obs['angle_to_point'][0])) - 10

        return reward

    def is_terminal_step(self, state):
        return True if calcs.get_2d_distance(state.players[0].car_data.position, self.point_position) < 200 else False

    def format_actions(self, actions, mode):
        if mode == 'step':
            return {'throttle': np.abs(actions[0]), 'steer': actions[1], 'pitch': 0.0, 'yaw': 0.0, 'roll': 0.0,
                    'jump': 0.0, 'boost': 0 < actions[2], 'handbrake': 0 < actions[3]}
        elif mode == 'send':
            return [1.0, actions['throttle'], actions['steer'], actions['pitch'], actions['yaw'], actions['roll'], float(actions['jump']), float(actions['boost']), float(actions['handbrake'])]
        elif mode == 'send_from_raw':
            actions = self.format_actions(actions, 'step')
            actions = self.format_actions(actions, 'send')
            return actions
