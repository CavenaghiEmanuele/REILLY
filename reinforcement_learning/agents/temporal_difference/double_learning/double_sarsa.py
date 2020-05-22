import numpy as np
from typing import List, Dict
from collections import defaultdict
from tqdm import tqdm

from ....structures import ActionValue, Policy
from ....environments.environment import Environment
from .double_temporal_difference import DoubleTemporalDifference


class DoubleSarsaAgent(DoubleTemporalDifference ,object):

    __slots__ = ['_A']

    def __repr__(self):
        return "DoubleSarsa: " + "alpha=" + str(self._alpha) + ", gamma=" + str(self._gamma) + ", epsilon=" + str(self._epsilon)

    def reset(self):
        self._episode_ended = False
        self._S = self._env.reset_env()
        policy_average = (self._policy[self._S] + self._policy2[self._S])/2
        self._A = np.random.choice(range(self._env.actions_size()), p=policy_average)
    
    def run_step(self, *args, **kwargs):
        n_S, R, self._episode_ended, _ = self._env.run_step(self._A, mod="train")
        policy_average = (self._policy[n_S] + self._policy2[n_S])/2
        n_A = np.random.choice(range(self._env.actions_size()), p=policy_average)

        if np.random.binomial(1, 0.5) == 0:
            self._Q[self._S, self._A] += self._alpha * (R + (self._gamma * self._Q2[n_S, n_A]) - self._Q[self._S, self._A])
            self._update_policy(self._S, self._policy, self._Q)
        else:
            self._Q2[self._S, self._A] += self._alpha * (R + (self._gamma * self._Q[n_S, n_A]) - self._Q2[self._S, self._A])
            self._update_policy(self._S, self._policy2, self._Q2)
        
        self._S = n_S
        self._A = n_A
