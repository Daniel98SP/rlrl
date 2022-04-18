import sys
from os.path import exists
from stable_baselines3 import PPO
from rocketleaguegym.gamelaunch.launch import launch_rocket_league
from rocketleaguegym.communication.communication_handler import CommunicationHandler
from training_environment import TrainingEnvironment
from predicting_environment import PredictingEnvironment

if __name__ == '__main__':
    game_process = launch_rocket_league(CommunicationHandler.format_pipe_id(0), 'epic')

    if sys.argv[1] == 'training':
        environment = TrainingEnvironment(game_process=game_process)

        if exists('models/model.zip'):
            model = PPO.load('models/model.zip', env=environment)
        else:
            model = PPO(policy='MultiInputPolicy', batch_size=64, n_epochs=32, env=environment, verbose=1)

        while True:
            model.learn(total_timesteps=10 * 2048, n_eval_episodes=10)
            model.save('models/model.zip')

    elif sys.argv[1] == 'predicting':
        environment = PredictingEnvironment(game_process=game_process)
        model = PPO.load('models/model.zip', env=environment)
        data_file = open('data/model_stats.txt', 'w')
        for _ in range(1000):
            observation = environment.reset()
            steps = 0
            done = False
            while not done:
                actions = model.predict(observation)[0]
                observation, done = environment.step(actions)
                steps = steps + 1
            data_file.write(str(steps/15) + '\n')
        data_file.close()
        environment.close()
