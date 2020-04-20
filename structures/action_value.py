import pandas as pd
import numpy as np


class Action_value():

    _action_value: pd.DataFrame

    def __init__(self, n_states, n_actions) -> None:
        self._action_value = pd.DataFrame(np.zeros(shape=(n_states, n_actions)))
    
    def __repr__(self):
        return str(self._action_value)
        