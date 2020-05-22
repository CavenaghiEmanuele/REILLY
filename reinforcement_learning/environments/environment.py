import gym

from abc import ABC, abstractmethod
from typing import List


class Environment(ABC):

    _env: gym.Env

    @abstractmethod
    def run_step(self, action, mod):
        pass

    @abstractmethod
    def render(self):
        pass

    @abstractmethod
    def reset_env(self):
        pass

    @abstractmethod
    def states_size(self) -> int:
        pass

    @abstractmethod
    def actions_size(self) -> int:
        pass

    @abstractmethod
    def get_env_tests(self) -> list:
        pass

    @abstractmethod
    def probability_distribution(self):
        pass
