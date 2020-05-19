from abc import ABC, abstractmethod


class Agent(ABC, object):

    __slots__ = ['_alpha', '_gamma', '_epsilon', '_S', '_episode_ended', '_Q', '_env', '_policy']

    @abstractmethod
    def run(self, n_episodes: int, n_tests: int, test_step: int):
        pass

    @abstractmethod
    def train(self, n_episodes: int) -> None:
        pass

    @abstractmethod
    def test(self, n_tests: int):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def run_step(self, *args, **kwargs):
        pass

    @abstractmethod
    def _control(self):
        pass
