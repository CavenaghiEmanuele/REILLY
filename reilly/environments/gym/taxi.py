import gym

from .abstract_gym import GymEnvironment


class Taxi(GymEnvironment):

    def __init__(self):
        self._env = gym.make('Taxi-v3')
        self.reset()

    def run_step(self, action, *args, **kwargs):
        next_state, reward, done, _ = self._env.step(action)
        info = {'wins': 0}
        if done and reward == 20:
            info['wins'] = 1
        return next_state, reward, done, info
