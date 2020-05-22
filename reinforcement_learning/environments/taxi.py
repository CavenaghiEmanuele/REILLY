import gym

from .environment import Environment

class Taxi(Environment):

    def __init__(self):
        self._env = gym.make("Taxi-v3")
        self.reset_env()

    def render(self):
        return self._env.render()
  
    # If mod flag is "test" return additional dict with environment tests result
    def run_step(self, action, mod: str):
        next_state, reward, done, _ = self._env.step(action)
        
        if mod == "test":
            if done and reward == 20:
                test_info = {"return_sum": reward, "wins": 1}
            else:
                test_info = {"return_sum": reward, "wins": 0}
            return next_state, reward, done, test_info

        return next_state, reward, done, _

    def reset_env(self):
        return self._env.reset()

    def states_size(self):
        return self._env.observation_space.n

    def actions_size(self):
        return self._env.action_space.n
    
    def get_env_tests(self):
        return ["return_sum", "wins"]

    def probability_distribution(self):
        return self._env.env.P
