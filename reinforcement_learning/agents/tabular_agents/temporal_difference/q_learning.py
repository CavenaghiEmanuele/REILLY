import numpy as np
from typing import List, Dict

from ....structures import ActionValue, Policy
from .temporal_difference import TemporalDifference


class QLearningAgent(TemporalDifference, object):

    def __repr__(self):
        return "QLearning: " + "alpha=" + str(self._alpha) + ", gamma=" + str(self._gamma) + ", epsilon=" + str(self._epsilon)
    
    def reset(self, env):
        self._episode_ended = False
        self._S = env.reset_env()
    
    def run_step(self, env, *args, **kwargs):
        A = np.random.choice(range(env.actions_size()), p=self._policy[self._S])
        n_S, R, self._episode_ended, info = env.run_step(A, **kwargs)
        self._Q[self._S, A] += self._alpha * (R + (self._gamma * np.max(self._Q[n_S])) - self._Q[self._S, A])
        self._update_policy(self._S)
        
        self._S = n_S

        return (n_S, R, self._episode_ended, info)
