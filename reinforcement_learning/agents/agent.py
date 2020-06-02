from abc import ABC, abstractmethod


class Agent(ABC, object):

    __slots__ = ['_alpha', '_gamma', '_epsilon', '_n_step', '_S', '_episode_ended', '_Q', '_env', '_policy']

    @abstractmethod
    def reset(self, env):
        pass

    @abstractmethod
    def run_step(self, env, *args, **kwargs):
        pass
