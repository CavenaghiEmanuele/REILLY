import gym

from ..environment import Environment

class Frozen_Lake4x4(Environment):

    def __init__(self, slippery=True):
        if slippery:
            self._env = gym.make("FrozenLake-v0")
        else:
            gym.register(
                id='FrozenLakeNotSlippery4x4-v0',
                entry_point='gym.envs.toy_text:FrozenLakeEnv',
                kwargs={'map_name' : '4x4', 'is_slippery': False},
                max_episode_steps=1000,
            )
            self._env = gym.make("FrozenLakeNotSlippery4x4-v0")
        self.reset_env()

    def render(self):
        return self._env.render()
  
    # If mod flag is "test" return additional dict with environment tests result
    def run_step(self, action, mod: str):
        next_state, reward, done, _ = self._env.step(action)
        
        if mod == "test":
            if done and reward == 1:
                test_info = {"wins": 1}
            else:
                test_info = {"wins": 0}      
            return next_state, reward, done, test_info

        return next_state, reward, done, _

    def reset_env(self):
        return self._env.reset()

    def states_size(self):
        return self._env.observation_space.n

    def actions_size(self):
        return self._env.action_space.n

    def get_env_tests(self):
        return ["wins"]

    def probability_distribution(self):
        return self._env.env.P


class Frozen_Lake8x8(Environment):

    def __init__(self, slippery=True):
        if slippery:
            self._env = gym.make("FrozenLake8x8-v0")
        else:
            gym.register(
                id='FrozenLakeNotSlippery8x8-v0',
                entry_point='gym.envs.toy_text:FrozenLakeEnv',
                kwargs={'map_name' : '4x4', 'is_slippery': False},
                max_episode_steps=1000,
            )
            self._env = gym.make("FrozenLakeNotSlippery8x8-v0")
        self.reset_env()

    def render(self):
        return self._env.render()
  
    # If mod flag is "test" return additional dict with environment tests result
    def run_step(self, action, mod: str):
        next_state, reward, done, _ = self._env.step(action)
        
        if mod == "test":
            if done and reward == 1:
                test_info = {"wins": 1}
            else:
                test_info = {"wins": 0}      
            return next_state, reward, done, test_info

        return next_state, reward, done, _

    def reset_env(self):
        return self._env.reset()

    def states_size(self):
        return self._env.observation_space.n

    def actions_size(self):
        return self._env.action_space.n
    
    def get_env_tests(self):
        return ["wins"]

    def probability_distribution(self):
        return self._env.env.P