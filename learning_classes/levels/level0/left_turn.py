from rocketleaguegym.communication.message import Message


class LeftTurn:
    def __init__(self, repetitions=1):
        self.number_of_ticks = 1
        self.number_of_repetitions = repetitions
        self.current_tick = 1
        self.current_repetition = 1

    def is_terminal_tick(self):
        done = False

        if self.current_repetition > self.number_of_repetitions and self.current_tick > self.number_of_ticks:
            done = True

        if self.current_tick > self.number_of_ticks:
            self.current_repetition += 1
            self.current_tick = 1
        else:
            self.current_tick += 1

        return done

    @staticmethod
    def execute_action(comms):
        pitch = yaw = roll = jump = boost = handbrake = 0.0
        throttle = 1.0
        steer = -1.0

        comms.send_message(header=Message.RLGYM_AGENT_ACTION_IMMEDIATE_RESPONSE_MESSAGE_HEADER, body=[1.0, throttle, steer, pitch, yaw, roll, jump, boost, handbrake])
