import gym

from ..environment import Environment


class MountainCar(Environment):

    def __init__(self):
        self._env = gym.make("MountainCar-v0")
        self.reset_env()

    def render(self):
        return self._env.render()

    # If mod flag is "test" return additional dict with environment tests result
    def run_step(self, action, *args, **kwargs):
        next_state, reward, done, _ = self._env.step(action)
        if kwargs['mode'] == "test":
            return next_state, reward, done, {"return_sum": reward}
        return next_state, reward, done, _

    def reset_env(self):
        return self._env.reset()

    def states_size(self):
        '''
        Discretize the state space for tabular agents. One simple way in which
        this can be done is to round the first element of the state vector to 
        the nearest 0.1 and the second element to the nearest 0.01, and then 
        (for convenience) multiply the first element by 10 and the second by 100.
        '''
        return 285

    def actions_size(self):
        return self._env.action_space.n

    def probability_distribution(self):
        return self._env.env.P
