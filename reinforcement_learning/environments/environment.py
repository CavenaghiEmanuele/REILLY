from abc import ABC, abstractmethod
from typing import List


class Environment(ABC):

    @property
    @abstractmethod
    def states_size(self) -> int:
        pass

    @property
    @abstractmethod
    def actions_size(self) -> int:
        pass

    @abstractmethod
    def render(self) -> None:
        pass

    @abstractmethod
    def reset(self) -> int:
        pass

    @abstractmethod
    def run_step(self, action, *args, **kwargs):
        pass

    @property
    @abstractmethod
    def probability_distribution(self):
        pass
