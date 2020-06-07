import numpy as np
from typing import List, Dict

from ...agent import Agent
from ....structures import ActionValue, Policy
from ..q_estimator import QEstimator


class TemporalDiffernceAppr(Agent, object):

    __slots__ = ["_Q_estimator"]

    def __init__(self, alpha, epsilon, gamma, feature_dims, num_tilings, tiling_offset=None, tiles_size=None):
        # self._policy  -> Approximante agents don't have policy but approximate it
        self._alpha = alpha
        self._epsilon = epsilon
        self._gamma = gamma
        self._Q_estimator = QEstimator(alpha=alpha,
                                       feature_dims=feature_dims,
                                       num_tilings=num_tilings,
                                       tiling_offset=tiling_offset,
                                       tiles_size=tiles_size
                                       )

    def _e_greedy_policy(self, env, state):
        n_actions = env.actions_size
        action_probs = np.zeros(n_actions)
        q_values = [self._Q_estimator.predict(state, action)
                    for action in range(n_actions)]
        indices = [i for i, x in enumerate(q_values) if x == max(q_values)]
        best_action = np.random.choice(indices)

        for action in range(n_actions):
            if action == best_action:
                action_probs[action] = 1 - self._epsilon + \
                    (self._epsilon/n_actions)
            else:
                action_probs[action] = self._epsilon/n_actions
        return action_probs
