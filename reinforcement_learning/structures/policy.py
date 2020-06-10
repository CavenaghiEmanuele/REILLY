import numpy as np


class Policy(object):

    __slots__ = ["_policy"]

    def __init__(self, n_states, n_actions) -> None:
        self._policy = np.ones(shape=(n_states, n_actions)) / n_actions

    def __repr__(self):
        return str(self._policy)

    def __getitem__(self, key):
        return self._policy[key]

    def __setitem__(self, key, value):
        self._policy[key] = value

    def __add__(self, other):
        return self._policy + other._policy

    def get_n_actions(self, state) -> int:
        return len(self._policy[state])
