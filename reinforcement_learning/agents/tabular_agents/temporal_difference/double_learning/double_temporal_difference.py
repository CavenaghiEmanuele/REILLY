import numpy as np
from typing import List, Dict

from .....structures import ActionValue, Policy
from ...temporal_difference import TemporalDifference


class DoubleTemporalDifference(TemporalDifference, object):

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

    def _update_policy(self, S: int, policy: Policy, Q: ActionValue) -> None:
        # Avoid choosing always the first move in case policy has the same value
        indices = [i for i, x in enumerate(Q[S]) if x == max(Q[S])]
        A_star = np.random.choice(indices)

        n_actions = policy.get_n_actions(S)
        for A in range(n_actions):
            if A == A_star:
                policy[S, A] = 1 - self._epsilon + (self._epsilon / n_actions)
            else:
                policy[S, A] = self._epsilon / n_actions
