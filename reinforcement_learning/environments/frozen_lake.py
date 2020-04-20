import gym

from .environment import Environment

class Frozen_Lake4x4(Environment):

    def __init__(self, slippery=True):
        if slippery:
            self._env = gym.make("FrozenLake-v0")
        else:
            self._env = gym.make("FrozenLakeNotSlippery4x4-v0")
        self.reset_env()

    def __repr__(self):
        return self._env.render()
  
    def run_step(self, action):
        return self._env.step(action)

    def reset_env(self):
        return self._env.reset()

    def get_state_number(self):
        return self._env.observation_space.n

    def get_action_number(self):
        return self._env.action_space.n

    def probability_distribution(self):
        return self._env.env.P


class Frozen_Lake8x8(Environment):

    def __init__(self, slippery=True):
        if slippery:
            self._env = gym.make("FrozenLake8x8-v0")
        else:
            self._env = gym.make("FrozenLakeNotSlippery8x8-v0")
        self.reset_env()

    def __repr__(self):
        return self._env.render()
  
    def run_step(self, action):
        return self._env.step(action)

    def reset_env(self):
        return self._env.reset()

    def get_state_number(self):
        return self._env.observation_space.n

    def get_action_number(self):
        return self._env.action_space.n

    def probability_distribution(self):
        return self._env.env.P