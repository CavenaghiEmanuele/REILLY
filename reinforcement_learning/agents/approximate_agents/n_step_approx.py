import numpy as np
from typing import List, Dict

from ..agent import Agent
from ...structures import ActionValue, Policy
from .q_estimator import QEstimator


class NStepApprox(Agent, object):

    __slots__ = ["_Q_estimator"]

    def __init__(self, alpha, epsilon, gamma, n_step, num_tilings, tiling_offset, tiles_dims, max_size=4096):
        # self._policy  -> Approximante agents don't have policy but approximate it
        self._alpha = alpha
        self._epsilon = epsilon
        self._gamma = gamma
        self._n_step = n_step
        self._Q_estimator = QEstimator(alpha=alpha,
                                       num_tilings=num_tilings,
                                       tiling_offset=tiling_offset,
                                       tiles_dims=tiles_dims,
                                       max_size=max_size
                                       )

    def _e_greedy_policy(self, env, state):

        n_actions = env.actions_size()
        action_probs = np.zeros(n_actions)
        q_values = [self._Q_estimator.predict(state, action)
                    for action in range(n_actions)]
        best_action = np.argmax(q_values)

        for action in range(n_actions):
            if action == best_action:
                action_probs[action] = 1 - self._epsilon + \
                    (self._epsilon/n_actions)
            else:
                action_probs[action] = self._epsilon/n_actions

        return action_probs
