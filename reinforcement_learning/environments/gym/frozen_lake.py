import gym

from .abstract_gym import GymEnvironment


class Frozen_Lake4x4(GymEnvironment):

    def __init__(self, slippery=True):
        if slippery:
            self._env = gym.make("FrozenLake-v0")
        else:
            gym.register(
                id='FrozenLakeNotSlippery4x4-v0',
                entry_point='gym.envs.toy_text:FrozenLakeEnv',
                kwargs={'map_name': '4x4', 'is_slippery': False},
                max_episode_steps=1000,
            )
            self._env = gym.make("FrozenLakeNotSlippery4x4-v0")
        self.reset()

    # If mod flag is "test" return additional dict with environment tests result
    def run_step(self, action, *args, **kwargs):
        next_state, reward, done, _ = self._env.step(action)

        if kwargs['mode'] == "test":
            info = {"wins": 0}
            if done and reward == 1:
                info["wins"] = 1
            return next_state, reward, done, info

        return next_state, reward, done, _


class Frozen_Lake8x8(GymEnvironment):

    def __init__(self, slippery=True):
        if slippery:
            self._env = gym.make("FrozenLake8x8-v0")
        else:
            gym.register(
                id='FrozenLakeNotSlippery8x8-v0',
                entry_point='gym.envs.toy_text:FrozenLakeEnv',
                kwargs={'map_name': '4x4', 'is_slippery': False},
                max_episode_steps=1000,
            )
            self._env = gym.make("FrozenLakeNotSlippery8x8-v0")
        self.reset()

    # If mod flag is "test" return additional dict with environment tests result
    def run_step(self, action, *args, **kwargs):
        next_state, reward, done, _ = self._env.step(action)

        if kwargs['mode'] == "test":
            info = {"wins": 0}
            if done and reward == 1:
                info["wins"] = 1
            return next_state, reward, done, info

        return next_state, reward, done, _
