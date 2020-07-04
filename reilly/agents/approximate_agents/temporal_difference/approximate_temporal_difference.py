import numpy as np
from abc import ABC

from ..approximate_agent import ApproximateAgent


class ApproximateTemporalDifference(ApproximateAgent, ABC, object):

    def reset(self, init_state:int, *args, **kwargs) -> None:
        self._S = init_state
        self._A = self._select_action(init_state)
