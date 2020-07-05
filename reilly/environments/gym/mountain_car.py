import gym

from .abstract_gym import GymEnvironment


class MountainCar(GymEnvironment):

    def __init__(self):
        self._env = gym.make("MountainCar-v0")
        self.reset()

    @property
    def states(self):
        '''
        Discretize the state space for tabular agents. One simple way in which
        this can be done is to round the first element of the state vector to
        the nearest 0.1 and the second element to the nearest 0.01, and then
        (for convenience) multiply the first element by 10 and the second by 100.
        '''
        return 285

    def run_step(self, action, *args, **kwargs):
        next_state, reward, done, _ = self._env.step(action)
        return next_state, reward, done, _
