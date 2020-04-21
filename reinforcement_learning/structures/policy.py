import pandas as pd
import numpy as np


class Policy():

    _policy: pd.DataFrame

    def __init__(self, n_states, n_actions) -> None:
        self._policy = pd.DataFrame(np.ones(shape=(n_states, n_actions))/n_actions)
    
    def __repr__(self):
        return str(self._policy)

    def __getitem__(self, key):
        return self._policy.loc[key]

    def __setitem__(self, key, value):
        self._policy.loc[key] = value

    def get_n_actions(self, state) -> int: 
        return len(self._policy.loc[state])