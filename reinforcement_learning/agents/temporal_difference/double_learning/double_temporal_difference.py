import numpy as np
from typing import List, Dict
from collections import defaultdict
from tqdm import tqdm
from abc import ABC, abstractmethod

from ....structures import ActionValue, Policy
from ....environments.environment import Environment
from ..temporal_difference import TemporalDifference


class DoubleTemporalDifference(TemporalDifference, ABC, object):

    __slots__ = ["_Q2", "_policy2"]

    def __init__(self, alpha, epsilon, gamma, environment):
        super().__init__(alpha, epsilon, gamma, environment)
        self._Q2 = ActionValue(environment.get_state_number(), environment.get_action_number())
        self._policy2 = Policy(environment.get_state_number(), environment.get_action_number())


    @abstractmethod
    def _control(self):
        pass
    
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
