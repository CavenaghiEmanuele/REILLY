import pandas as pd
import numpy as np


class State_value():

    _state_value: pd.Series

    def __init__(self, n_states) -> None:
        self._state_value = pd.Series(np.zeros(n_states))

    
    def __repr__(self):
        return str(self._state_value)
