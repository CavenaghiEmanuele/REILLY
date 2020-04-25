import numpy as np
from typing import List, Dict
from collections import defaultdict
from tqdm import tqdm

from ..structures import Action_value, Policy
from ..environments.environment import Environment
from .temporal_difference import TemporalDifferenceAgent


class SarsaAgent(TemporalDifferenceAgent, object):

    def __init__(self, alpha, epsilon, gamma, environment):
        super().__init__(alpha, epsilon, gamma, environment)
    
    def __repr__(self):
        return "Sarsa: " + "alpha=" + str(self._alpha) + ", gamma=" + str(self._gamma) + ", epsilon=" + str(self._epsilon)
    
    def _control(self):
        episode_ended = False
        S = self._env.reset_env()
        A = np.random.choice(range(self._env.get_action_number()), p=self._policy[S])

        while not episode_ended:
            n_S, R, episode_ended, _ = self._env.run_step(A, "train")
            n_A = np.random.choice(range(self._env.get_action_number()), p=self._policy[n_S])

            self._Q[S, A] += self._alpha * (R + (self._gamma * self._Q[n_S, n_A]) - self._Q[S, A]) 
            self._update_policy(S)
            S = n_S
            A = n_A
  