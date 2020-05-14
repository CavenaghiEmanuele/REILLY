import numpy as np
from typing import List, Dict
from collections import defaultdict
from tqdm import tqdm

from ....structures import ActionValue, Policy
from ....environments.environment import Environment
from .double_temporal_difference import DoubleTemporalDifference


class DoubleQLearningAgent(DoubleTemporalDifference ,object):

    def __init__(self, alpha, epsilon, gamma, environment):
        super().__init__(alpha, epsilon, gamma, environment)

    def __repr__(self):
        return "DoubleQLearning: " + "alpha=" + str(self._alpha) + ", gamma=" + str(self._gamma) + ", epsilon=" + str(self._epsilon)

    def _control(self):
        episode_ended = False
        S = self._env.reset_env()

        while not episode_ended:
            policy_average = (self._policy[S] + self._policy2[S])/2
            A = np.random.choice(range(self._env.get_action_number()), p=policy_average)
            n_S, R, episode_ended, _ = self._env.run_step(A, "train")
            
            if np.random.binomial(1, 0.5) == 0:
                self._Q[S, A] += self._alpha * (R + (self._gamma * self._Q2[n_S, np.argmax(self._Q[n_S])]) - self._Q[S, A])
                self._update_policy(S, self._policy, self._Q)
            else:
                self._Q2[S, A] += self._alpha * (R + (self._gamma * self._Q[n_S, np.argmax(self._Q2[n_S])]) - self._Q2[S, A])
                self._update_policy(S, self._policy2, self._Q2)

            S = n_S
