from rocketleaguegym.communication.message import Message


class FrontFlip:
    def __init__(self, repetitions=1):
        self.number_of_ticks = 180
        self.number_of_repetitions = repetitions
        self.current_tick = 1
        self.current_repetition = 1

    def is_terminal_tick(self):
        return self.current_tick > self.number_of_ticks

    def execute_action(self, comms):
        throttle = steer = pitch = yaw = roll = jump = boost = handbrake = 0.0

        if 1 <= self.current_tick <= 10:
            jump = 1.0
        elif self.current_tick == 20:
            jump = 1.0
            pitch = -1.0

        self.current_tick += 1
        comms.send_message(header=Message.RLGYM_AGENT_ACTION_IMMEDIATE_RESPONSE_MESSAGE_HEADER, body=[1.0, throttle, steer, pitch, yaw, roll, jump, boost, handbrake])
