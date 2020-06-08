import gym

from ..environment import Environment


class MountainCar(Environment):

    def __init__(self):
        self._env = gym.make("MountainCar-v0")
        self.reset()

    @property
    def states_size(self):
        '''
        Discretize the state space for tabular agents. One simple way in which
        this can be done is to round the first element of the state vector to
        the nearest 0.1 and the second element to the nearest 0.01, and then
        (for convenience) multiply the first element by 10 and the second by 100.
        '''
        return 285

    @property
    def actions_size(self):
        return self._env.action_space.n

    def render(self):
        return self._env.render()

    def reset(self) -> int:
        return self._env.reset()

    # If mod flag is "test" return additional dict with environment tests result
    def run_step(self, action, *args, **kwargs):
        next_state, reward, done, _ = self._env.step(action)
        if kwargs['mode'] == "test":
            return next_state, reward, done, {"return_sum": reward}
        return next_state, reward, done, _

    @property
    def probability_distribution(self):
        return self._env.env.P
