from abc import ABC, abstractmethod


class Environment(ABC):

    @property
    @abstractmethod
    def states(self) -> int:
        pass

    @property
    @abstractmethod
    def actions(self) -> int:
        pass

    @abstractmethod
    def render(self) -> None:
        pass

    @abstractmethod
    def reset(self, *args, **kwargs) -> int:
        pass

    @abstractmethod
    def run_step(self, action, *args, **kwargs):
        pass

    @property
    @abstractmethod
    def probability_distribution(self):
        pass
