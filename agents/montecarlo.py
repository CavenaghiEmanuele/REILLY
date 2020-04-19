from ..structures import state_value


class MonteCarloAgent():

    _epsilon: int
    _gamma: int
    _state_value: state_value.State_value

    def __init__(self, epsilon, gamma):
        self._epsilon = epsilon
        self._gamma = gamma