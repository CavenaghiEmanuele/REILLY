import gym

from .abstract_gym import GymEnvironment


class FrozenLake4x4(GymEnvironment):

    def __init__(self, slippery=True):
        if slippery:
            self._env = gym.make('FrozenLake-v0')
        else:
            gym.register(
                id='FrozenLakeNotSlippery4x4-v0',
                entry_point='gym.envs.toy_text:FrozenLakeEnv',
                kwargs={'map_name': '4x4', 'is_slippery': False},
                max_episode_steps=1000,
            )
            self._env = gym.make('FrozenLakeNotSlippery4x4-v0')
        self.reset()

    def run_step(self, action, *args, **kwargs):
        next_state, reward, done, _ = self._env.step(action)
        info = {'wins': 0}
        if done and reward == 1:
            info['wins'] = 1
        return next_state, reward, done, info


class FrozenLake8x8(GymEnvironment):

    def __init__(self, slippery=True):
        if slippery:
            self._env = gym.make('FrozenLake8x8-v0')
        else:
            gym.register(
                id='FrozenLakeNotSlippery8x8-v0',
                entry_point='gym.envs.toy_text:FrozenLakeEnv',
                kwargs={'map_name': '4x4', 'is_slippery': False},
                max_episode_steps=1000,
            )
            self._env = gym.make('FrozenLakeNotSlippery8x8-v0')
        self.reset()

    def run_step(self, action, *args, **kwargs):
        next_state, reward, done, _ = self._env.step(action)
        info = {'wins': 0}
        if done and reward == 1:
            info['wins'] = 1
        return next_state, reward, done, info
