import numpy as np

from typing import List
from abc import ABC, abstractmethod

from .. import backend

class Agent():

    __slots__ = [
        '_alpha', '_gamma', '_epsilon', '_e_decay', '_n_step',
        '_A', '_S', '_episode_ended', '_actions',
    ]

    _A: int

    def __new__(cls, *args, **kwargs):
        if kwargs.get('backend', None) == 'cpp':
            params = {k: v for k, v in kwargs.items() if k != 'backend'}
            instan = getattr(backend, cls.__name__)
            if instan is None:
                raise NotImplementedError()
            return instan(*args, **params)
        return super(Agent, cls).__new__(cls)

    def get_action(self):
        return self._A

    @abstractmethod
    def update(self, n_S: int, R: float, done: bool, *args, **kwargs):
        pass

    @abstractmethod
    def reset(self, env, *args, **kwargs):
        pass

    @abstractmethod
    def _select_action(self, weights: List) -> None:
        pass
