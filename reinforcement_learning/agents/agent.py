from abc import ABC, abstractmethod


class Agent(ABC, object):

    __slot__ = ['_alpha', '_gamma', '_epsilon']

    @abstractmethod
    def _control(self):
        pass
