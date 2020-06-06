from ..environment import Environment


class GymEnvironment(Environment):

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

    @property
    def probability_distribution(self):
        return self._env.env.P
