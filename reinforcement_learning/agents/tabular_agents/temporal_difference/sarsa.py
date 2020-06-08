import numpy as np
from typing import List, Dict

from ....structures import ActionValue, Policy

from .temporal_difference import TemporalDifference


class SarsaAgent(TemporalDifference, object):

    __slots__ = ['_A']

    def __repr__(self):
        return "Sarsa: " + "alpha=" + str(self._alpha) + ", gamma=" + \
            str(self._gamma) + ", epsilon=" + str(self._epsilon)

    def reset(self, env):
        self._episode_ended = False
        self._S = env.reset()
        self._A = np.random.choice(range(env.actions_size), p=self._policy[self._S])

    def run_step(self, env, *args, **kwargs):
        n_S, R, self._episode_ended, info = env.run_step(self._A, **kwargs)
        n_A = np.random.choice(range(env.actions_size), p=self._policy[n_S])

        self._Q[self._S, self._A] += self._alpha * (R + (self._gamma * self._Q[n_S, n_A]) - self._Q[self._S, self._A])
        self._update_policy(self._S)

        self._S = n_S
        self._A = n_A

        return (n_S, R, self._episode_ended, info)
