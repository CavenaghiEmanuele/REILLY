from abc import ABC, abstractmethod


class Agent(ABC, object):

    __slots__ = ['_alpha', '_gamma', '_epsilon', '_S', '_episode_ended', '_Q', '_env', '_policy']

    @abstractmethod
    def _control(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def run_step(self, *args, **kwargs):
        pass
