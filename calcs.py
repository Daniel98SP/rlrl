import math
import numpy as np

import values
from rocketleaguegym.gamestates.game_state import GameState


# General functions
def get_2d_distance(position1, position2) -> float:
    return np.sqrt(np.power(position2[0] - position1[0], 2) + np.power(position2[1] - position1[1], 2))


def get_3d_distance(position1, position2) -> float:
    return np.sqrt(np.power(position2[0] - position1[0], 2) + np.power(position2[1] - position1[1], 2) + np.power(position2[2] - position1[2], 2))


def get_2d_vector(position1, position2) -> np.ndarray:
    return np.array([position2[0] - position1[0], position2[1] - position1[1]])


def get_3d_vector(position1, position2) -> np.ndarray:
    return np.array([position2[0] - position1[0], position2[1] - position1[1], position2[2] - position1[2]])


def get_angle_between_vectors(vector1, vector2) -> float:
    return np.rad2deg(math.atan2(vector1[0] * vector2[1] - vector1[1] * vector2[0], vector1[0] * vector2[0] + vector1[1] * vector2[1]))


def get_vector_from_rotation(theta):
    return np.dot(np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]), [1, 0])


# def rotate_vector(vector, angle, input_angle_type='degrees') -> np.ndarray:
    # theta = np.deg2rad(angle) if input_angle_type == 'degrees' else angle
    # rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    # rotated_vector = np.dot(rotation, vector)

    # return rotated_vector

# def angle_to_origin_vector(input, input_type):
    # if input_type == 'rotation':
        # return np.rad2deg(input)
    # elif input_type == 'vector':
        # return get_angle_between_vectors([1, 0], input)


# Observation related functions
def get_velocity_angle_to_point(object1_position, object1_velocity_vector, object2_position):
    object_1_to_2_vector = get_2d_vector(object1_position, object2_position)
    velocity_angle_to_point = - get_angle_between_vectors(object1_velocity_vector, object_1_to_2_vector)

    return 180 if object1_velocity_vector[0] == 0.0 and object1_velocity_vector[1] == 0.0 else velocity_angle_to_point


def get_yaw_angle_to_point(state: GameState, point_position):
    car_yaw_vector = get_vector_from_rotation(state.players[0].car_data.yaw())
    car_to_point_vector = get_2d_vector(state.players[0].car_data.position[0:2], point_position)
    car_angle_to_point = get_angle_between_vectors(car_to_point_vector, car_yaw_vector)
    return car_angle_to_point


def get_speed(state: GameState):
    x_velocity = state.players[0].car_data.linear_velocity[0]
    y_velocity = state.players[0].car_data.linear_velocity[1]
    return np.sqrt(np.power(x_velocity, 2) + np.power(y_velocity, 2))

def get_point_behind_ball(state, distance):
    goal_position = values.ORANGE_GOAL_CENTER if state.players[0].team_num == values.BLUE_TEAM else values.BLUE_GOAL_CENTER

    goal_to_ball_vector = get_2d_vector(goal_position, state.ball.position)
    resized_goal_to_ball_vector = goal_to_ball_vector * (distance / np.linalg.norm(goal_to_ball_vector))

    return [state.ball.position[0] + resized_goal_to_ball_vector[0] + state.ball.linear_velocity[0], state.ball.position[1] + resized_goal_to_ball_vector[1], state.ball.position[2]]
