import numpy as np
from typing import List, Dict
from collections import defaultdict
from tqdm import tqdm

from ..structures import Action_value, Policy
from ..environments.environment import Environment
from .temporal_difference import TemporalDifferenceAgent


class DoubleQLearningAgent(TemporalDifferenceAgent ,object):

    __slots__ = ["_Q2", "_policy2"]

    def __init__(self, alpha, epsilon, gamma, environment):
        super().__init__(alpha, epsilon, gamma, environment)
        self._Q2 = Action_value(environment.get_state_number(), environment.get_action_number())
        self._policy2 = Policy(environment.get_state_number(), environment.get_action_number())


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
    
    def _update_policy(self, S, policy, Q) -> None:
        # Avoid choosing always the first move in case policy has the same value
        indices = [i for i, x in enumerate(Q[S]) if x == max(Q[S])]
        A_star = np.random.choice(indices)

        n_actions = policy.get_n_actions(S)
        for A in range(n_actions):
            if A == A_star:
                policy[S, A] = 1 - self._epsilon + (self._epsilon/n_actions)
            else:
                policy[S, A] = self._epsilon/n_actions
