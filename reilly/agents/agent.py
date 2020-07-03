import numpy as np

from typing import List
from abc import ABC, abstractmethod


class Agent(ABC, object):

    __slots__ = ['_alpha', '_gamma', '_epsilon', '_e_decay', '_n_step',
                 '_A', '_S', '_episode_ended', '_actions',
                 '_Q', '_policy']

    def get_action(self):
        return self._A
    
    def _select_action(self, weights: List) -> None:
        return np.random.choice(range(self._actions), p=weights)

    def _e_greedy_policy(self, weights: List) -> None:
        # Avoid choosing always the first move in case policy has the same value
        indices = [i for i, x in enumerate(weights) if x == max(weights)]
        A_star = np.random.choice(indices)

        for A in range(self._actions):
            if A == A_star:
                self._policy[self._S, A] = 1 - self._epsilon + (self._epsilon / self._actions)
            else:
                self._policy[self._S, A] = self._epsilon / self._actions

    @abstractmethod
    def reset(self, env, *args, **kwargs):
        pass
