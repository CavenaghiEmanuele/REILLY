import numpy as np
from typing import List, Dict
from collections import defaultdict
from tqdm import tqdm
import itertools

from ..agent import Agent
from ...structures import ActionValue, Policy
from ...environments.environment import Environment


class NStep(Agent, object):

    def __init__(self, alpha, epsilon, gamma, n_step, environment):
        self._env = environment
        n_states = self._env.get_state_number()
        n_actions = self._env.get_action_number()
        self._Q = ActionValue(n_states, n_actions)
        self._policy = Policy(n_states, n_actions)
        self._alpha = alpha
        self._epsilon = epsilon
        self._gamma = gamma
        self._n_step = n_step


    def run(self, n_episodes: int, n_tests: int, test_step: int):
        test_results = defaultdict(list)
        for n_episode in tqdm(range(n_episodes)):
            self._control()
            if (n_episode % test_step) == 0:
                test_info = self.test(n_tests)
                [test_results[test].append(test_info[test]) for test in test_info.keys()]
        return test_results 

    def train(self, n_episodes: int) -> None:
        for _ in tqdm(range(n_episodes)):
            self._control()

    #Repeat test n_tests time to avoid outliers results
    def test(self, n_tests: int):
        test_results = defaultdict(list)
        for _ in range(n_tests):
            test_info = self._play_episode()            
            [test_results[test].append(test_info[test]) for test in test_info.keys()]
        return {test : np.average(test_results[test]) for test in test_results.keys()}

    def _control(self):
        self.reset()
        for t in itertools.count():
            if self.run_step(t = t):
                break

    def _update_policy(self, S) -> None:
        # Avoid choosing always the first move in case policy has the same value
        indices = [i for i, x in enumerate(self._Q[S]) if x == max(self._Q[S])]
        A_star = np.random.choice(indices)

        n_actions = self._policy.get_n_actions(S)
        for A in range(n_actions):
            if A == A_star:
                self._policy[S, A] = 1 - self._epsilon + (self._epsilon/n_actions)
            else:
                self._policy[S, A] = self._epsilon/n_actions

    # n-step bootstrapping agent's use play_episode function only for tests
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