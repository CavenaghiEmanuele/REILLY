import random

from ..structures import Action_value, Policy
from ..environments.environment import Environment


class MonteCarloAgent():

    _epsilon: int
    _gamma: int
    _action_value: Action_value
    _policy: Policy
    _env: Environment

    def __init__(self, epsilon, gamma, environment):
        self._env = environment
        n_states = self._env.get_state_number()
        n_actions = self._env.get_action_number()
        self._action_value = Action_value(n_states, n_actions)
        self._policy = Policy(n_states, n_actions)

        self._epsilon = epsilon
        self._gamma = gamma

    
    def monteCarlo_prediction(self, n_episodes):
        
        returns = {}

        for _ in range(n_episodes):
            episode_trajectory = self.play_episode()
            G = 0
            
            for i in reversed(range(len(episode_trajectory))):
                s_t, a_t, r_t = episode_trajectory[i]
                G = (G * self._gamma) + r_t #Update expected return

                #First visit implementation
                if not (s_t, a_t) in [(s[0], s[1]) for s in episode_trajectory[0:i]]:
                    returns.setdefault((s_t, a_t), 0)
                    returns[(s_t, a_t)] += 1
                    self._action_value[s_t, a_t] += (1 / (i+1)) * (G - self._action_value[s_t, a_t])



    def play_episode(self):
        state = self._env.reset_env()
        episode_ended = False
        episode_trajectory = []
        
        while not episode_ended:
            #CHANGE ACTION SELECTION
            action = random.choice(range(self._env.get_action_number()))
            next_state, reward, episode_ended, _ = self._env.run_step(action)
            episode_trajectory.append((state, action, reward))
            state = next_state

        return episode_trajectory
