import numpy as np
from typing import List, Dict
from collections import defaultdict
from tqdm import tqdm

from ...structures import ActionValue, Policy
from ...environments.environment import Environment
from .temporal_difference import TemporalDifference


class QLearningAgent(TemporalDifference ,object):

    def __repr__(self):
        return "QLearning: " + "alpha=" + str(self._alpha) + ", gamma=" + str(self._gamma) + ", epsilon=" + str(self._epsilon)
    
    def reset(self):
        self._episode_ended = False
        self._S = self._env.reset_env()
    
    def run_step(self):
        A = np.random.choice(range(self._env.get_action_number()), p=self._policy[self._S])
        n_S, R, self._episode_ended, _ = self._env.run_step(A, "train")
        self._Q[self._S, A] += self._alpha * (R + (self._gamma * np.max(self._Q[n_S])) - self._Q[self._S, A])
        self._update_policy(self._S)
        
        self._S = n_S
