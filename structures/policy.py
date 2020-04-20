import pandas as pd
import numpy as np


class Policy():

    _policy: pd.DataFrame

    def __init__(self, n_states, n_actions) -> None:
        self._policy = pd.DataFrame(np.ones(shape=(n_states, n_actions))/n_actions)
    
    def __repr__(self):
        return str(self._policy)
