import numpy as np

from ..tabular_agent import TabularAgent
from ....structures import ActionValue, Policy


class NStep(TabularAgent, object):
    
    __slots__ = ['_states', '_action_list', '_rewards', 'T']

    def __init__(self, 
                states:int,
                actions:int, 
                alpha:float, 
                epsilon:float, 
                gamma:float, 
                n_step:int,
                epsilon_decay:float = 1):
        self._Q = ActionValue(states, actions)
        self._policy = Policy(states, actions)
        self._alpha = alpha
        self._epsilon = epsilon
        self._gamma = gamma
        self._n_step = n_step
        self._e_decay = epsilon_decay
        self._actions = actions

    def reset(self, init_state: int, *args, **kwargs) -> None:
        self._states = [init_state]
        self._action_list = [self._select_action(self._policy[init_state])]
        self._rewards = [0.0]
        self.T = float('inf')
        
    def get_action(self):
        return self._action_list[-1]