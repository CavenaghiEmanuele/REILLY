import numpy as np
from typing import List, Dict
from collections import defaultdict
from tqdm import tqdm

from ..structures import Action_value, Policy
from ..environments.environment import Environment


class SarsaAgent(object):

    __slots__ = ["_alpha", "_epsilon", "_gamma", "_Q", "_policy", "_env"]

    def __init__(self, alpha, epsilon, gamma, environment):
        #Basic attribute
        self._env = environment
        n_states = self._env.get_state_number()
        n_actions = self._env.get_action_number()
        self._Q = Action_value(n_states, n_actions)
        self._policy = Policy(n_states, n_actions)
        self._alpha = alpha
        self._epsilon = epsilon
        self._gamma = gamma
    

    def run(self, n_episodes: int, n_tests: int, test_step: int):
        test_results = defaultdict(list)
        for n_episode in tqdm(range(n_episodes)):
            self._sarsa_control()
            if (n_episode % test_step) == 0:
                test_info = self.test(n_tests)
                [test_results[test].append(test_info[test]) for test in test_info.keys()]
        return test_results 

    def train(self, n_episodes: int) -> None:
        for _ in tqdm(range(n_episodes)):
            self._sarsa_control()

    #Repeat test n_tests time to avoid outliers results
    def test(self, n_tests: int):
        test_results = defaultdict(list)
        for _ in tqdm(range(n_tests)):
            test_info = self._play_episode()            
            [test_results[test].append(test_info[test]) for test in test_info.keys()]
        return {test : np.average(test_results[test]) for test in test_results.keys()}
    
    def _sarsa_control(self):

        episode_ended = False
        state = self._env.reset_env()
        action = np.random.choice(range(self._env.get_action_number()), p=self._policy[state])

        while not episode_ended:
            next_state, reward, episode_ended, _ = self._env.run_step(action, "train")
            next_action = np.random.choice(range(self._env.get_action_number()), p=self._policy[next_state])

            self._Q[state, action] += self._alpha * (reward + (self._gamma * self._Q[next_state, next_action]) - self._Q[state, action]) 
            self._update_policy(state)
            state = next_state
            action = next_action
        
    def _update_policy(self, state_t) -> None:
        # Avoid choosing always the first move in case policy has the same value
        indices = [i for i, x in enumerate(self._Q[state_t]) if x == max(self._Q[state_t])]
        a_star = np.random.choice(indices)

        n_actions = self._policy.get_n_actions(state_t)
        for action in range(n_actions):
            if action == a_star:
                self._policy[state_t, action] = 1 - self._epsilon + (self._epsilon/n_actions)
            else:
                self._policy[state_t, action] = self._epsilon/n_actions

    # Sarsa agent's use play_episode function only for tests
    def _play_episode(self) -> List:
        state = self._env.reset_env()
        episode_ended = False
        test_results = defaultdict(float)

        while not episode_ended:
            #Select action according to policy distribution probability
            action = np.random.choice(range(self._env.get_action_number()), p=self._policy[state])
            next_state, _, episode_ended, test_info = self._env.run_step(action, "test")
            state = next_state
            for test in test_info.keys():
                test_results[test] += test_info[test]
        return test_results
