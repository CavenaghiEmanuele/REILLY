import numpy as np


class State_value():

    __slots__=  ["_state_value"]

    def __init__(self, n_states) -> None:
        self._state_value = np.zeros(n_states)

    def __repr__(self):
        return str(self._state_value)

    def __getitem__(self, key):
        return self._state_value[key]

    def __setitem__(self, key, value):
        self._state_value[key] = value
