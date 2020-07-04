from typing import List

from .approximate_temporal_difference import ApproximateTemporalDifference


class SemiGradientExpectedSarsa(ApproximateTemporalDifference, object):

    def __repr__(self):
        return "SemiGradient Expected Sarsa: " + "alpha=" + str(self._alpha) + \
            ", gamma=" + str(self._gamma) + \
            ", epsilon=" + str(self._epsilon) + \
            ", e-decay=" + str(self._e_decay)

    def update(self, n_S: int, R: float, done: bool, *args, **kwargs) -> None:

        if done and kwargs['training']:
            self._Q_estimator.update(self._S, self._A, R)
            return 

        n_A = self._select_action(n_S)

        if kwargs['training']:
            G = R + (self._gamma * self._compute_expected_value(n_S))
            self._Q_estimator.update(self._S, self._A, G)

        self._S = n_S
        self._A = n_A
        
        if done: self._epsilon *= self._e_decay


    def _compute_expected_value(self, state: List) -> float:
        expected_value = 0
        for action in range(self._actions):
            expected_value += self._e_greedy_policy(state)[action] * self._Q_estimator.predict(state, action)
        return expected_value