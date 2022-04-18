from rocketleaguegym.gamestates.game_state import GameState
from rocketleaguegym.gamestates.wrappers.physics_wrapper import PhysicsWrapper
from rocketleaguegym.gamestates.wrappers.car_wrapper import CarWrapper
from values import BLUE_TEAM, ORANGE_TEAM
from typing import List


class StateWrapper(object):
    BLUE_ID1 = 1
    ORANGE_ID1 = 5

    def __init__(self, blue_count: int = 0, orange_count: int = 0, game_state=None):
        if game_state is None:
            self.ball: PhysicsWrapper = PhysicsWrapper()
            self.cars: List[CarWrapper] = []
            for i in range(blue_count):
                self.cars.append(CarWrapper(BLUE_TEAM, StateWrapper.BLUE_ID1 + i))
            for i in range(orange_count):
                self.cars.append(CarWrapper(ORANGE_TEAM, StateWrapper.ORANGE_ID1 + i))
        else:
            self._read_from_gamestate(game_state)

    def _read_from_gamestate(self, game_state: GameState):
        self.ball: PhysicsWrapper = PhysicsWrapper(game_state.ball)
        self.cars: List[CarWrapper] = []
        for player in game_state.players:
            self.cars.append(CarWrapper(player_data=player))

    def format_state(self) -> list:
        ball_state = self.ball.encode()

        car_states = []
        for car in self.cars:
            car_states += car.encode()

        encoded = ball_state + car_states

        return encoded
