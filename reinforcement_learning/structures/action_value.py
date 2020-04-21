import pandas as pd
import numpy as np


class Action_value():

    _action_value: pd.DataFrame

    def __init__(self, n_states, n_actions) -> None:
        self._action_value = pd.DataFrame(np.zeros(shape=(n_states, n_actions)))
    
    def __repr__(self):
        return str(self._action_value)

    def __getitem__(self, key):
        return self._action_value.loc[key]

    def __setitem__(self, key, value):
        self._action_value.loc[key] = value

    def argmax(self):
        return self._action_value.idxmax(axis=1)
        