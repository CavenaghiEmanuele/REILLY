import numpy as np
from typing import List, Tuple

from .....structures import ActionValue, Policy
from .....environments import Environment
from .double_temporal_difference import DoubleTemporalDifference


class DoubleExpectedSarsa(DoubleTemporalDifference, object):

    __slots__ = ['_A']

    def __repr__(self):
        return "DoubleExpectedSarsa: " + "alpha=" + str(self._alpha) + \
        ", gamma=" + str(self._gamma) + \
        ", epsilon=" + str(self._epsilon) + \
        ", e-decay=" + str(self._e_decay)

    def update(self, n_S: int, R: float, done: bool, *args, **kwargs) -> Tuple:
        n_A = self._select_action((self._policy[n_S] + self._policy2[n_S]) / 2)

        if kwargs['training']:
            if np.random.binomial(1, 0.5) == 0:
                self._Q[self._S, self._A] += self._alpha * \
                    (R + (self._gamma * self._compute_expected_value(self._policy2, self._Q2, n_S)) -
                    self._Q[self._S, self._A])
                self._policy_update(self._S, self._policy, self._Q)
            else:
                self._Q2[self._S, self._A] += self._alpha * \
                    (R + (self._gamma * self._compute_expected_value(self._policy, self._Q, n_S)) -
                    self._Q2[self._S, self._A])
                self._policy_update(self._S, self._policy2, self._Q2)

        self._S = n_S
        self._A = n_A
        
        if done: 
            self._epsilon *= self._e_decay
    
    def _compute_expected_value(self, policy: Policy, Q: ActionValue, state: int) -> float:
        expected_value = 0
        for action in range(len(Q[state])):
            expected_value += policy[state, action] * Q[state, action]
        return expected_value
