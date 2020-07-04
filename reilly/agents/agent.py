import numpy as np

from typing import List
from abc import ABC, abstractmethod


class Agent(ABC, object):

    __slots__ = ['_alpha', '_gamma', '_epsilon', '_e_decay', '_n_step',
                 '_A', '_S', '_episode_ended', '_actions',
                ]

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
