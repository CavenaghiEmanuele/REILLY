from abc import abstractmethod

from ..environment import Environment


class GymEnvironment(Environment):

    _env: Environment

    @property
    def states_size(self) -> int:
        return self._env.observation_space.n

    @property
    def actions_size(self) -> int:
        return self._env.action_space.n

    def render(self):
        return self._env.render()

    def reset(self) -> int:
        return self._env.reset()

    @abstractmethod
    def run_step(self, action, *args, **kwargs):
        pass

    @property
    def probability_distribution(self):
        return self._env.env.P
