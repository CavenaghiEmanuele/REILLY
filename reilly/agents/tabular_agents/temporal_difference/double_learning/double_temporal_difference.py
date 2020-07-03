from abc import ABC

from .....structures import ActionValue, Policy
from ...temporal_difference import TemporalDifference


class DoubleTemporalDifference(TemporalDifference, ABC, object):

    __slots__ = ["_Q2", "_policy2"]

    def __init__(self, 
                states:int, 
                actions:int, 
                alpha:float, 
                epsilon:float, 
                gamma:float,
                epsilon_decay:float = 1):
        super().__init__(states, actions, alpha, epsilon, gamma, epsilon_decay)
        self._Q2 = ActionValue(states, actions)
        self._policy2 = Policy(states, actions)

