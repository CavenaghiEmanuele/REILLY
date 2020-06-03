import gym

from ..environment import Environment

class Taxi(Environment):

    def __init__(self):
        self._env = gym.make("Taxi-v3")
        self.reset()
    
    def states_size(self) -> int:
        return self._env.observation_space.n

    def actions_size(self) -> int:
        return self._env.action_space.n

    def render(self):
        return self._env.render()
    
    def reset(self) -> int:
        return self._env.reset()
  
    # If mod flag is "test" return additional dict with environment tests result
    def run_step(self, action, *args, **kwargs):
        next_state, reward, done, _ = self._env.step(action)
        
        if kwargs['mode'] == "test":
            if done and reward == 20:
                info = {"return_sum": reward, "wins": 1}
            else:
                info = {"return_sum": reward, "wins": 0}
            return next_state, reward, done, info

        return next_state, reward, done, _

    def probability_distribution(self):
        return self._env.env.P
