import numpy as np
from typing import List, Dict
from collections import defaultdict
from tqdm import tqdm

from ..structures import Action_value, Policy
from ..environments.environment import Environment
from .temporal_difference import TemporalDifference


class ExpectedSarsaAgent(TemporalDifference, object):

    def __init__(self, alpha, epsilon, gamma, environment):
        super().__init__(alpha, epsilon, gamma, environment)

    def __repr__(self):
        return "ExpectedSarsa: " + "alpha=" + str(self._alpha) + ", gamma=" + str(self._gamma) + ", epsilon=" + str(self._epsilon)

    def _control(self):
        episode_ended = False
        S = self._env.reset_env()
        A = np.random.choice(range(self._env.get_action_number()), p=self._policy[S])

        while not episode_ended:
            n_S, R, episode_ended, _ = self._env.run_step(A, "train")
            n_A = np.random.choice(range(self._env.get_action_number()), p=self._policy[n_S])

            self._Q[S, A] += self._alpha * (R + (self._gamma * self._compute_expected_value(n_S)) - self._Q[S, A]) 
            self._update_policy(S)
            S = n_S
            A = n_A

    def _compute_expected_value(self, state) -> float:
        expected_value = 0
        for action in range(len(self._Q[state])):
            expected_value += self._policy[state, action] * self._Q[state, action]
        return expected_value
