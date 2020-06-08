import numpy as np
from typing import List, Dict

from .....structures import ActionValue, Policy
from ...temporal_difference import TemporalDifference


class DoubleTemporalDifference(TemporalDifference, object):

    __slots__ = ["_Q2", "_policy2"]

    def __init__(self, states_size, actions_size, alpha, epsilon, gamma):
        super().__init__(states_size, actions_size, alpha, epsilon, gamma)
        self._Q2 = ActionValue(states_size, actions_size)
        self._policy2 = Policy(states_size, actions_size)

    def _update_policy(self, S, policy, Q) -> None:
        # Avoid choosing always the first move in case policy has the same value
        indices = [i for i, x in enumerate(Q[S]) if x == max(Q[S])]
        A_star = np.random.choice(indices)

        n_actions = policy.get_n_actions(S)
        for A in range(n_actions):
            if A == A_star:
                policy[S, A] = 1 - self._epsilon + (self._epsilon / n_actions)
            else:
                policy[S, A] = self._epsilon / n_actions
