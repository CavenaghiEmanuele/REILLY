from abc import ABC

from ..tabular_agent import TabularAgent


class TemporalDifference(TabularAgent, ABC, object):
    
    def reset(self, init_state:int, *args, **kwargs) -> None:
        self._S = init_state
        self._A = self._select_action(self._policy[init_state])
