import numpy as np
from typing import List, Tuple

from ....environments import Environment
from .temporal_difference_appr import TemporalDiffernceAppr


class SarsaApproximateAgent(TemporalDiffernceAppr, object):

    __slots__ = ['_A']

    def __repr__(self):
        return "Sarsa Appr: " + "alpha=" + str(self._alpha) + \
            ", gamma=" + str(self._gamma) + \
            ", epsilon=" + str(self._epsilon) + \
            ", e-decay=" + str(self._e_decay)

    def reset(self, env: Environment, *args, **kwargs) -> None:
        self._episode_ended = False
        self._S = env.reset(*args, **kwargs)
        self._A = np.random.choice(
            range(env.actions_size), p=self._e_greedy_policy(self._S, env.actions_size))

    def run_step(self, env: Environment, *args, **kwargs) -> Tuple:
        n_S, R, self._episode_ended, info = env.run_step(self._A, **kwargs)

        if self._episode_ended and not kwargs['mode'] == "test":
            self._Q_estimator.update(self._S, self._A, R)
            return (n_S, R, self._episode_ended, info)

        n_A = np.random.choice(range(env.actions_size),
                               p=self._e_greedy_policy(n_S, env.actions_size))

        if not kwargs['mode'] == "test":
            G = R + (self._gamma * self._Q_estimator.predict(n_S, n_A))
            self._Q_estimator.update(self._S, self._A, G)

        self._S = n_S
        self._A = n_A
        
        if self._episode_ended:
            self._epsilon *= self._e_decay
        return (n_S, R, self._episode_ended, info)
