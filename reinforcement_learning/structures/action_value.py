import numpy as np
  

class Action_value(object):

    __slots__ = ["_action_value"]

    def __init__(self, n_states, n_actions) -> None:
        self._action_value = np.zeros(shape=(n_states, n_actions))
    
    def __repr__(self):
        return str(self._action_value)

    def __getitem__(self, key):
        return self._action_value[key]

    def __setitem__(self, key, value):
        self._action_value[key] = value

    def argmax(self):
        return self._action_value.argmax(axis=0)
    
    def __add__(self, other):
        return self._action_value + other._action_value
