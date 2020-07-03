import numpy as np

from .temporal_difference import TemporalDifference


class ExpectedSarsa(TemporalDifference, object):

    def __repr__(self):
        return "ExpectedSarsa: " + "alpha=" + str(self._alpha) + \
            ", gamma=" + str(self._gamma) + \
            ", epsilon=" + str(self._epsilon) + \
            ", e-decay=" + str(self._e_decay)

    def update(self, n_S: int, R: float, done: bool, *args, **kwargs) -> None:
        n_A = self._select_action(self._policy[n_S])
        
        if kwargs['training']:
            self._Q[self._S, self._A] += self._alpha * \
                (R + (self._gamma * self._expected_value(n_S)) - self._Q[self._S, self._A])
            self._policy_update(self._S, self._policy, self._Q)

        self._S = n_S
        self._A = n_A
        
        if done: self._epsilon *= self._e_decay
    
    def _expected_value(self, state: int) -> float:
        expected_value = 0
        for action in range(len(self._Q[state])):
            expected_value += self._policy[state, action] * self._Q[state, action]
        return expected_value
