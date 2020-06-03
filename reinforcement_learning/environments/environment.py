from abc import ABC, abstractmethod
from typing import List


class Environment(ABC):

    @abstractmethod
    def states_size(self) -> int:
        pass

    @abstractmethod
    def actions_size(self) -> int:
        pass

    @abstractmethod
    def render(self):
        pass

    @abstractmethod
    def reset_env(self) -> int:
        pass

    @abstractmethod
    def run_step(self, action, *args, **kwargs):
        pass

    @abstractmethod
    def probability_distribution(self):
        pass
