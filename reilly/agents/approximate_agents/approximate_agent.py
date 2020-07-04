import numpy as np
from abc import ABC
from typing import List

from ..agent import Agent
from .q_estimator import QEstimator


class ApproximateAgent(Agent, ABC, object):

    __slots__ = ["_Q_estimator"]

    def __init__(
        self,
        actions: int,
        alpha: float,
        epsilon: float,
        gamma: float,
        features: int,
        tilings: int,
        epsilon_decay: float = 1,
        tilings_offset: List = None,
        tile_size: List = None,
        *args,
        **kwargs
    ):
        self._actions = actions
        self._alpha = alpha
        self._epsilon = epsilon
        self._gamma = gamma
        self._e_decay = epsilon_decay
        self._Q_estimator = QEstimator(alpha=alpha,
                                       feature_dims=features,
                                       num_tilings=tilings,
                                       tiling_offset=tilings_offset,
                                       tiles_size=tile_size
                                       )

    def _select_action(self, state: List) -> None:
        return np.random.choice(range(self._actions), p=self._e_greedy_policy(state))

    def _e_greedy_policy(self, state: List) -> List:
        action_probs = np.zeros(self._actions)
        q_values = [self._Q_estimator.predict(state, action)
                    for action in range(self._actions)]
        indices = [i for i, x in enumerate(q_values) if x == max(q_values)]
        best_action = np.random.choice(indices)

        for action in range(self._actions):
            if action == best_action:
                action_probs[action] = 1 - self._epsilon + \
                    (self._epsilon / self._actions)
            else:
                action_probs[action] = self._epsilon / self._actions
        return action_probs
