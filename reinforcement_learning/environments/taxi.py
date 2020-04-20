import gym

from .environment import Environment

class Taxi(Environment):

    def __init__(self):
        self._env = gym.make("Taxi-v2")
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
