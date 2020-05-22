import numpy as np
from typing import List, Dict
from collections import defaultdict
from tqdm import tqdm
import itertools

from ..agent import Agent
from ...structures import ActionValue, Policy


class NStep(Agent, object):

    def __init__(self, states_size, actions_size, alpha, epsilon, gamma, n_step):
        self._Q = ActionValue(states_size, actions_size)
        self._policy = Policy(states_size, actions_size)
        self._alpha = alpha
        self._epsilon = epsilon
        self._gamma = gamma
        self._n_step = n_step

    def _update_policy(self, S) -> None:
        # Avoid choosing always the first move in case policy has the same value
        indices = [i for i, x in enumerate(self._Q[S]) if x == max(self._Q[S])]
        A_star = np.random.choice(indices)

        n_actions = self._policy.get_n_actions(S)
        for A in range(n_actions):
            if A == A_star:
                self._policy[S, A] = 1 - self._epsilon + \
                    (self._epsilon/n_actions)
            else:
                self._policy[S, A] = self._epsilon/n_actions
