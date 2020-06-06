import numpy as np
from typing import List, Dict

from ....structures import ActionValue, Policy
from .temporal_difference import TemporalDifference


class ExpectedSarsaAgent(TemporalDifference, object):

    __slots__ = ['_A']

    def __repr__(self):
        return "ExpectedSarsa: " + "alpha=" + str(self._alpha) + ", gamma=" + str(self._gamma) + ", epsilon=" + str(self._epsilon)

    def _compute_expected_value(self, state) -> float:
        expected_value = 0
        for action in range(len(self._Q[state])):
            expected_value += self._policy[state, action] * self._Q[state, action]
        return expected_value
    
    def reset(self, env):
        self._episode_ended = False
        self._S = env.reset()
        self._A = np.random.choice(range(env.actions_size), p=self._policy[self._S])
    
    def run_step(self, env, *args, **kwargs):
        n_S, R, self._episode_ended, info = env.run_step(self._A, **kwargs)
        n_A = np.random.choice(range(env.actions_size), p=self._policy[n_S])

        self._Q[self._S, self._A] += self._alpha * (R + (self._gamma * self._compute_expected_value(n_S)) - self._Q[self._S, self._A]) 
        self._update_policy(self._S)

        self._S = n_S
        self._A = n_A

        return (n_S, R, self._episode_ended, info)
