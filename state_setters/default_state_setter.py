import values
from rocketleaguegym.gamestates.wrappers.state_wrapper import StateWrapper
import numpy as np
from numpy import random
import calcs


class StateSetter:
    @staticmethod
    def reset(state_wrapper: StateWrapper, cars_position='random', cars_yaw='random', cars_speed='zero', ball_position='random', ball_speed='zero'):
        StateSetter._reset_cars(state_wrapper, cars_position, cars_yaw, cars_speed)
        StateSetter._reset_ball(state_wrapper, ball_position, ball_speed)

    @staticmethod
    def _reset_cars(state_wrapper, cars_position, cars_yaw, cars_speed):
        for car in state_wrapper.cars:
            index = random.choice([0, 1, 2, 3, 4])

            # Boost
            car.boost = random.random()

            # Position
            if cars_position == 'random':
                car.position = [random.uniform(-3072.0, 3072.0), random.uniform(-4096.0, 4096.0), 17.0]
            elif cars_position == 'default':
                if car.team_num == values.BLUE_TEAM:
                    car.position = [[-2048, -2560, 17], [2048, -2560, 17], [-256, -3840, 17], [256, -3840, 17], [0, -4608, 17]][index]
                elif car.team_num == values.ORANGE_TEAM:
                    car.position = [[2048, 2560, 17], [-2048, 2560, 17], [256, 3840, 17], [-256, 3840, 17], [0, 4608, 17]][index]
            elif cars_position == 'inverted':
                if car.team_num == values.BLUE_TEAM:
                    car.position = [[2048, 2560, 17], [-2048, 2560, 17], [256, 3840, 17], [-256, 3840, 17], [0, 4608, 17]][index]
                elif car.team_num == values.ORANGE_TEAM:
                    car.position = [[-2048, -2560, 17], [2048, -2560, 17], [-256, -3840, 17], [256, -3840, 17], [0, -4608, 17]][index]
            else:
                car.position = cars_position

            # Yaw
            if cars_position == 'random':
                car.rotation = [0.0, random.uniform(-np.pi, np.pi), 0.0]
            elif cars_position == 'default':
                if car.team_num == values.BLUE_TEAM:
                    car.rotation = [0.25 * np.pi, 0.75 * np.pi, 0.5 * np.pi, 0.5 * np.pi, 0.5 * np.pi][index]
                elif car.team_num == values.ORANGE_TEAM:
                    car.rotation = [-0.75 * np.pi, -0.25 * np.pi, -0.5 * np.pi, -0.5 * np.pi, -0.5 * np.pi][index]
            elif cars_position == 'inverted':
                if car.team_num == values.BLUE_TEAM:
                    car.rotation = [-0.75 * np.pi, -0.25 * np.pi, -0.5 * np.pi, -0.5 * np.pi, -0.5 * np.pi][index]
                elif car.team_num == values.ORANGE_TEAM:
                    car.rotation = [0.25 * np.pi, 0.75 * np.pi, 0.5 * np.pi, 0.5 * np.pi, 0.5 * np.pi][index]
            else:
                car.rotation = [0.0, cars_yaw, 0.0]

            # Speed
            if cars_speed == 'random':
                car_yaw_vector = calcs.get_vector_from_rotation(car.rotation[1])
                car_yaw_unit_vector = car_yaw_vector / np.linalg.norm(car_yaw_vector)
                speed = random.uniform(0.0, 2300.0)
                car.linear_velocity = [car_yaw_unit_vector[0] * speed, car_yaw_unit_vector[1] * speed, 0.0]

    @staticmethod
    def _reset_ball(state_wrapper, ball_position, ball_speed):
        # Position
        if ball_position == 'random':
            state_wrapper.ball.position = [random.uniform(-3072.0, 3072.0), random.uniform(-4096.0, 4096.0), 93.0]
        elif ball_position == 'outside':
            state_wrapper.ball.position = [80000.0, 80000.0, 93.0]
        else:
            state_wrapper.ball.position = ball_position

        # Speed
        if ball_speed == 'random':
            velocity_direction = random.rand(2)
            unit_velocity_direction = velocity_direction / np.linalg.norm(velocity_direction)
            speed = random.uniform(0.0, 1500.0)
            state_wrapper.ball.linear_velocity = [unit_velocity_direction[0] * speed, unit_velocity_direction[1] * speed, 0.0]
            # state_wrapper.ball.linear_velocity = [random.uniform(0.0, 1000.0), random.uniform(0.0, 1000.0), 0.0]
