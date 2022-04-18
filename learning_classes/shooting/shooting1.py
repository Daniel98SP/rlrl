import calcs

import numpy as np
from gym import spaces

import values
from rocketleaguegym.gamestates.game_state import GameState


class SHOOTING1:
    def __init__(self):
        self.steps = 0

    @staticmethod
    def get_spaces():
        observation_space_dictionary = dict()
        observation_space_dictionary['car_angle_to_ball'] = spaces.Box(low=-180.0, high=180.0, shape=(1,), dtype=np.float32)
        observation_space_dictionary['car_angle_to_goal'] = spaces.Box(low=-180.0, high=180.0, shape=(1,), dtype=np.float32)
        observation_space_dictionary['ball_velocity_angle_to_goal'] = spaces.Box(low=-180.0, high=180.0, shape=(1,), dtype=np.float32)
        observation_space_dictionary['horizontal_distance_to_ball'] = spaces.Box(low=0.0, high=1000.0, shape=(1,), dtype=np.float32)
        observation_space_dictionary['vertical_distance_to_ball'] = spaces.Box(low=0.0, high=2000.0, shape=(1,), dtype=np.float32)
        observation_space_dictionary['car_speed'] = spaces.Box(low=0.0, high=2300.0, shape=(1,), dtype=np.float32)
        # observation_space_dictionary['ball_velocity_relative_to_car'] = spaces.Box(low=0.0, high=6000.0, shape=(3,), dtype=np.float32)

        return spaces.Dict(observation_space_dictionary), spaces.Box(low=np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0]), high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), dtype=np.float32)

    @staticmethod
    def get_observation(state: GameState):
        goal_position = values.ORANGE_GOAL_CENTER if state.players[0].team_num == values.BLUE_TEAM else values.BLUE_GOAL_CENTER

        observation = dict()
        observation['car_angle_to_ball'] = [calcs.get_yaw_angle_to_point(state, state.ball.position)]
        observation['car_angle_to_goal'] = [calcs.get_yaw_angle_to_point(state, goal_position)]
        observation['ball_velocity_angle_to_goal'] = [calcs.get_velocity_angle_to_point(state.ball.position, state.ball.linear_velocity, goal_position)]
        observation['horizontal_distance_to_ball'] = [calcs.get_2d_distance(state.players[0].car_data.position, state.ball.position)]
        observation['vertical_distance_to_ball'] = [state.ball.position[2] - state.players[0].car_data.position[2]]
        observation['car_speed'] = [calcs.get_speed(state)]
        # observation['ball_speed_relative_to_car'] =

        return observation

    def get_reward(self, previous_obs, actions, current_obs):
        reward = 2000 if np.abs(current_obs['ball_velocity_angle_to_goal'][0]) < 10 else 0
        # reward += - np.sqrt(4 * np.abs(current_obs['car_angle_to_ball'][0]))
        # reward += - np.sqrt(4 * np.abs(current_obs['car_angle_to_goal'][0]))

        return reward

    def is_terminal_step(self, current_obs):
        self.steps += 1

        return True if np.abs(current_obs['ball_velocity_angle_to_goal'][0]) < 10 or self.steps == 360 else False

    def format_actions(self, actions, mode):
        if mode == 'step':
            return {'throttle': np.abs(actions[0]), 'steer': actions[1], 'pitch': actions[2], 'yaw': 0.0, 'roll': 0.0,
                    'jump': -0.05 < actions[3] < 0.05, 'boost': 0 < actions[4], 'handbrake': 0 < actions[5]}
        elif mode == 'send':
            return [1.0, actions['throttle'], actions['steer'], actions['pitch'], actions['yaw'], actions['roll'], float(actions['jump']), float(actions['boost']), float(actions['handbrake'])]
        elif mode == 'send_from_raw':
            actions = self.format_actions(actions, 'step')
            actions = self.format_actions(actions, 'send')
            return actions
