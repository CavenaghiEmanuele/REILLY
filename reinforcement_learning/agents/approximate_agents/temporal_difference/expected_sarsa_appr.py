import numpy as np
from typing import List, Tuple

from ....environments import Environment
from .temporal_difference_appr import TemporalDiffernceAppr


class ExpectedSarsaApproximateAgent(TemporalDiffernceAppr, object):

    __slots__ = ['_A']

    def __repr__(self):
        return "Expected Sarsa Appr: " + "alpha=" + str(self._alpha) + \
            ", gamma=" + str(self._gamma) + \
            ", epsilon=" + str(self._epsilon) + \
            ", e-decay=" + str(self._e_decay)

    def reset(self, env: Environment, *args, **kwargs) -> None:
        self._episode_ended = False
        self._S = env.reset()
        self._A = np.random.choice(
            range(env.actions), p=self._e_greedy_policy(self._S, env.actions))

    def run_step(self, env: Environment, *args, **kwargs) -> Tuple:

        n_S, R, self._episode_ended, info = env.run_step(self._A, **kwargs)

        if self._episode_ended and not kwargs['mode'] == "test":
            self._Q_estimator.update(self._S, self._A, R)
            return (n_S, R, self._episode_ended, info)

        n_A = np.random.choice(range(env.actions),
                               p=self._e_greedy_policy(n_S, env.actions))

        if not kwargs['mode'] == "test":
            G = R + (self._gamma * self._compute_expected_value(n_S, env.actions))
            self._Q_estimator.update(self._S, self._A, G)

        self._S = n_S
        self._A = n_A
        
        if self._episode_ended:
            self._epsilon *= self._e_decay
        return (n_S, R, self._episode_ended, info)

    def _compute_expected_value(self, state: List, n_action: int) -> float:
        expected_value = 0
        for action in range(n_action):
            expected_value += self._e_greedy_policy(state, n_action)[action] * self._Q_estimator.predict(state, action)
        return expected_value