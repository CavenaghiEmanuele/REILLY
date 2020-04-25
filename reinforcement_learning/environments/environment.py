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
    def get_state_number(self) -> int:
        pass

    @abstractmethod
    def get_action_number(self) -> int:
        pass

    @abstractmethod
    def probability_distribution(self):
        pass
