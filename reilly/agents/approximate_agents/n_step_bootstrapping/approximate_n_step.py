import numpy as np
from typing import List
from abc import ABC

from ..approximate_agent import ApproximateAgent
from ..q_estimator import QEstimator


class ApproximateNStep(ApproximateAgent, ABC, object):

    __slots__ = ['_states', '_action_list', '_rewards', 'T']

    def __init__(self,
                 actions: int,
                 alpha: float,
                 epsilon: float,
                 gamma: float,
                 n_step: int,
                 features: int,
                 tilings: int,
                 epsilon_decay: float = 1,
                 tilings_offset: List = None,
                 tile_size: List = None):
        
        self._actions = actions
        self._alpha = alpha
        self._epsilon = epsilon
        self._gamma = gamma
        self._n_step = n_step
        self._e_decay = epsilon_decay
        self._Q_estimator = QEstimator(alpha=alpha,
                                       feature_dims=features,
                                       num_tilings=tilings,
                                       tiling_offset=tilings_offset,
                                       tiles_size=tile_size
                                       )
        
    def reset(self, init_state: int, *args, **kwargs) -> None:
        self._states = [init_state]
        self._action_list = [self._select_action(init_state)]
        self._rewards = [0.0]
        self.T = float('inf')
    
    def get_action(self):
        return self._action_list[-1]
    