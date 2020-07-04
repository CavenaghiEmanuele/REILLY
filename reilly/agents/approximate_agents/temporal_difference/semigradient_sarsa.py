from typing import List

from .approximate_temporal_difference import ApproximateTemporalDifference


class SemiGradientSarsa(ApproximateTemporalDifference, object):

    def __repr__(self):
        return "SemiGradient Sarsa: " + "alpha=" + str(self._alpha) + \
            ", gamma=" + str(self._gamma) + \
            ", epsilon=" + str(self._epsilon) + \
            ", e-decay=" + str(self._e_decay)

    def update(self, n_S: int, R: float, done: bool, *args, **kwargs) -> None:

        if done and kwargs['training']:
            self._Q_estimator.update(self._S, self._A, R)
            return

        n_A = self._select_action(n_S)

        if kwargs['training']:
            G = R + (self._gamma * self._Q_estimator.predict(n_S, n_A))
            self._Q_estimator.update(self._S, self._A, G)

        self._S = n_S
        self._A = n_A
        
        if done: 
            self._epsilon *= self._e_decay
