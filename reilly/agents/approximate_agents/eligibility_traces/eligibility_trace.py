from typing import List
from abc import ABC

from ..approximate_agent import ApproximateAgent
from ..q_estimator import QEstimator


class EligibilityTrace(ApproximateAgent, ABC, object):

    __slots__ = ['_lambda']

    def __init__(self,
                 actions: int,
                 alpha: float,
                 epsilon: float,
                 gamma: float,
                 lambd: float,
                 features: int,
                 tilings: int,
                 epsilon_decay: float = 1,
                 trace_type: str = "replacing",
                 tilings_offset: List = None,
                 tile_size: List = None):

        self._actions = actions
        self._alpha = alpha
        self._epsilon = epsilon
        self._gamma = gamma
        self._lambda = lambd
        self._e_decay = epsilon_decay
        self._Q_estimator = QEstimator(alpha=alpha,
                                       feature_dims=features,
                                       num_tilings=tilings,
                                       tiling_offset=tilings_offset,
                                       tiles_size=tile_size,
                                       have_trace=True,
                                       trace_type=trace_type
                                       )

        
    def reset(self, init_state: int, *args, **kwargs) -> None:
        self._S = init_state
        self._A = self._select_action(init_state)
        self._Q_estimator.reset_traces()
        