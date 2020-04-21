import pandas as pd
import numpy as np


class State_value():

    _state_value: pd.Series

    def __init__(self, n_states) -> None:
        self._state_value = pd.Series(np.zeros(n_states))

    def __repr__(self):
        return str(self._state_value)

    def __getitem__(self, key):
        return self._state_value.loc[key]

    def __setitem__(self, key, value):
        self._state_value.loc[key] = value
