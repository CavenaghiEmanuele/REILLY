import numpy as np
from typing import List, Tuple

from ....environments import Environment
from .approximate_n_step import ApproximateNStep


class SemiGradientNStepExpectedSarsa(ApproximateNStep, object):

    def __repr__(self):
        return "SemiGradient n-step Expected Sarsa: " + "alpha=" + str(self._alpha) + \
            ", gamma=" + str(self._gamma) + \
            ", epsilon=" + str(self._epsilon) + \
            ", n-step=" + str(self._n_step) + \
            ", e-decay=" + str(self._e_decay)

    def update(self, n_S: int, R: float, done: bool, *args, **kwargs) -> None:
        t = kwargs['t']
        if t < self.T:
            self._states.append(n_S)
            self._rewards.append(R)

            if done:
                self.T = t + 1
            else:
                self._action_list.append(self._select_action(n_S))
                self._epsilon *= self._e_decay
        
        if kwargs['training']:
            pi = t - self._n_step + 1
            if pi >= 0:
                G = 0

                for i in range(pi + 1, min(self.T, pi + self._n_step) + 1):
                    G += np.power(self._gamma, i - pi - 1) * self._rewards[i]

                if pi + self._n_step < self.T:
                    G += np.power(self._gamma, self._n_step) * self._compute_expected_value(
                        self._states[pi + self._n_step])

                self._Q_estimator.update(self._states[pi], self._action_list[pi], G)

            
    def _compute_expected_value(self, state: List) -> float:
        expected_value = 0
        for action in range(self._actions):
            expected_value += self._e_greedy_policy(
                state)[action] * self._Q_estimator.predict(state, action)
        return expected_value
    