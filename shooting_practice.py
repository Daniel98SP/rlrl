import calcs
import values
from rocketleaguegym.gamestates.wrappers.state_wrapper import StateWrapper
import numpy as np
from numpy import random


class StateSetter:
    @staticmethod
    def reset(state_wrapper: StateWrapper):
        StateSetter._reset_ball_random(state_wrapper)
        StateSetter._reset_cars_random(state_wrapper)

    @staticmethod
    def _reset_ball_random(state_wrapper: StateWrapper):
        state_wrapper.ball.position = [random.uniform(-1, 1) * (values.SIDE_WALL_X - 500), random.uniform(-1, 1) * (values.BACK_WALL_Y - 500), 93.0]
        # state_wrapper.ball.linear_velocity = [random.uniform(0.0, 1100.0), random.uniform(0.0, 1100.0), 0.0]
        # state_wrapper.ball.set_angular_velocity(*rand_vec3(6))

    @staticmethod
    def _reset_cars_random(state_wrapper):
        for car in state_wrapper.cars:
            goal_position = values.ORANGE_GOAL_CENTER

            # Car position
            goal_to_ball_vector = calcs.get_2d_vector(goal_position, state_wrapper.ball.position)
            resized_goal_to_ball_vector = goal_to_ball_vector * (500 / np.linalg.norm(goal_to_ball_vector))
            car.position = [state_wrapper.ball.position[0] + resized_goal_to_ball_vector[0], state_wrapper.ball.position[1] + resized_goal_to_ball_vector[1],  17.0]

            # Car rotation
            player_to_goal_vector = calcs.get_2d_vector(car.position, goal_position)
            angle = calcs.get_angle_between_vectors([1, 0, 0], player_to_goal_vector)
            car.rotation = np.array([0.0, random.uniform(-0.349066, 0.349066) + np.deg2rad(angle), 0.0])

            # Car linear velocity
            car_to_ball_vector = calcs.get_2d_vector(state_wrapper.cars[0].position, state_wrapper.ball.position)
            car_to_ball_unit_vector = car_to_ball_vector / np.linalg.norm(car_to_ball_vector)
            speed = random.uniform(0.0, 2300.0)
            car.linear_velocity = [car_to_ball_unit_vector[0] * speed, car_to_ball_unit_vector[1] * speed, 0.0]

            # Car boost
            car.boost = random.random()
