import numpy as np

from typing import List, Dict
from collections import defaultdict

from .tile_coding import TileCoding


class QEstimator():
    """
    Linear action-value (q-value) function approximator for
    semi-gradient methods with state-action featurization via tile coding.
    """

    _tile_coding: TileCoding
    _num_tilings: int
    _alpha: float

    # Every tiling have a separated list with its weights and (optionally) traces
    _weights: List[defaultdict]
    _traces: np.array
    _have_trace: bool

    _trace_type: str  # accumulating or replacing

    def __init__(self, alpha: float, feature_dims: int, num_tilings: int,
                 tiling_offset: List[float], tiles_size: List[float], 
                 trace: bool = False, trace_type: str = "replacing"):

        self._tile_coding = TileCoding(
            feature_dims, tiling_offset, tiles_size, num_tilings)
        self._num_tilings = num_tilings
        # The learning rate alpha is scaled by number of tilings
        self._alpha = alpha / num_tilings
        self._weights = [defaultdict(lambda:0) for _ in range(num_tilings)]
        self._have_trace = trace
        # If trace is True initialize traces
        if self._have_trace:
            self._traces = [defaultdict(lambda:0) for _ in range(num_tilings)]
            self._trace_type = trace_type

    def predict(self, state: List, action=None, number_action=None):
        """
        Predicts q-value(s) using linear FA. If action a is given then returns prediction
        for single state-action pair (s, a). Otherwise returns predictions for all actions
        in environment paired with s.
        """
        if isinstance(state, np.ndarray):
            state.tolist()
        if isinstance(state, np.int64) or isinstance(state, int):
            state = [state]
        if action is None and number_action is None:
            raise "ERROR: one of action and number_action must be set"

        if action is None:
            features = [self._tile_coding.get_coordinates(state, i)
                        for i in range(number_action)]
        else:
            features = [self._tile_coding.get_coordinates(state, action)]

        value = 0
        for feature in features:
            for i in range(self._num_tilings):
                value += self._weights[i][feature[i]]
        return value

    def update(self, state, action, target):
        """
        Updates the estimator parameters for a given state and action towards
        the target using the gradient update rule (and the eligibility trace if one has been set).
        """
        if isinstance(state, np.ndarray):
            state.tolist()
        if isinstance(state, np.int64) or isinstance(state, int):
            state = [state]

        features = self._tile_coding.get_coordinates(state, action)
        estimation = sum([self._weights[i][features[i]]
                          for i in range(self._num_tilings)])  # Linear FA
        delta = target - estimation

        if self._have_trace:
            for i in range(self._num_tilings):
                if self._trace_type == "replacing":
                    self._traces[i][features[i]] = 1  # Replacing trace
                elif self._trace_type == "accumulating":
                    self._traces[i][features[i]] += 1  # Accumulating trace

                self._weights[i][features[i]] += self._alpha * \
                    delta * self._traces[i][features[i]]

        else:
            for i in range(self._num_tilings):
                self._weights[i][features[i]] += self._alpha * delta

    def reset_traces(self) -> None:
        """
        Resets the eligibility trace (must be done at the start of every epoch)
        """
        if self._have_trace:
            self._traces = [defaultdict(lambda:0)
                            for _ in range(self._num_tilings)]
