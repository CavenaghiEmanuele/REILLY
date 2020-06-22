import numpy as np
from typing import List

from ...agent import Agent
from ....structures import ActionValue, Policy
from ..q_estimator import QEstimator


class NStepAppr(Agent, object):

    __slots__ = ["_Q_estimator", '_states', '_actions', '_rewards', 'T']

    def __init__(self,
                 alpha: float,
                 epsilon: float,
                 gamma: float,
                 n_step: int,
                 feature_dims: int,
                 num_tilings: int,
                 epsilon_decay: float = 1,
                 tiling_offset: List = None,
                 tiles_size: List = None):
        # self._policy  -> Approximante agents don't have policy but approximate it
        self._alpha = alpha
        self._epsilon = epsilon
        self._gamma = gamma
        self._n_step = n_step
        self._e_decay = epsilon_decay
        self._Q_estimator = QEstimator(alpha=alpha,
                                       feature_dims=feature_dims,
                                       num_tilings=num_tilings,
                                       tiling_offset=tiling_offset,
                                       tiles_size=tiles_size
                                       )

    def _e_greedy_policy(self, state, n_actions):
        action_probs = np.zeros(n_actions)
        q_values = [self._Q_estimator.predict(state, action)
                    for action in range(n_actions)]
        indices = [i for i, x in enumerate(q_values) if x == max(q_values)]
        best_action = np.random.choice(indices)

        for action in range(n_actions):
            if action == best_action:
                action_probs[action] = 1 - self._epsilon + \
                    (self._epsilon / n_actions)
            else:
                action_probs[action] = self._epsilon / n_actions
        return action_probs
