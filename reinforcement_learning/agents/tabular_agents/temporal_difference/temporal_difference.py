import numpy as np
from typing import List, Dict
from abc import ABC, abstractmethod

from ...agent import Agent
from ....structures import ActionValue, Policy


class TemporalDifference(Agent, object):

    def __init__(self, 
                states:int, 
                actions:int, 
                alpha:float, 
                epsilon:float, 
                gamma:float,
                epsilon_decay:float = 1):
        self._Q = ActionValue(states, actions)
        self._policy = Policy(states, actions)
        self._alpha = alpha
        self._epsilon = epsilon
        self._gamma = gamma
        self._e_decay = epsilon_decay

    def _update_policy(self, S: int) -> None:
        # Avoid choosing always the first move in case policy has the same value
        indices = [i for i, x in enumerate(self._Q[S]) if x == max(self._Q[S])]
        A_star = np.random.choice(indices)

        n_actions = self._policy.get_n_actions(S)
        for A in range(n_actions):
            if A == A_star:
                self._policy[S, A] = 1 - self._epsilon + (self._epsilon / n_actions)
            else:
                self._policy[S, A] = self._epsilon / n_actions
