import numpy as np
from typing import List, Dict
from collections import defaultdict
from tqdm import tqdm

from .agent import Agent
from ..structures import ActionValue, Policy
from ..environments.environment import Environment


class MonteCarloAgent(Agent):

    __slots__ = ["_visit_update", "_policy_method", "_returns", '_episode_trajectory', '_test_results']

    def __init__(self, epsilon, gamma, environment, visit_update="first", policy_method="on-policy"):
        #Basic attribute
        self._env = environment
        n_states = self._env.states_size()
        n_actions = self._env.actions_size()
        self._Q = ActionValue(n_states, n_actions)
        self._policy = Policy(n_states, n_actions)
        self._returns = np.zeros(shape=(n_states, n_actions))
        self._epsilon = epsilon
        self._gamma = gamma

        #Flags
        self._visit_update = visit_update
        self._policy_method = policy_method

    def __repr__(self):
        return "MonteCarlo: " + "gamma=" + str(self._gamma) + ", epsilon=" + str(self._epsilon)

    def run(self, n_episodes: int, n_tests: int, test_step: int):
        test_results = defaultdict(list)
        for n_episode in tqdm(range(n_episodes)):
            self._control()
            if (n_episode % test_step) == 0:
                test_info = self.test(n_tests)
                for test in test_info.keys():
                    test_results[test].append(test_info[test])
        return test_results

    def train(self, n_episodes: int) -> None:
        for _ in tqdm(range(n_episodes)):
            self._control()
    
    #Repeat test n_tests time to avoid outliers results
    def test(self, n_tests: int):
        test_results = defaultdict(list)
        for _ in range(n_tests):
            test_info = self._play_episode(mod="test")
            for test in test_info.keys():
                test_results[test].append(test_info[test])
        return {test : np.average(test_results[test]) for test in test_results.keys()}
   
    def _control(self) -> None:
        self.reset()
        while not self._episode_ended:
            self.run_step(mod='train')

    def _first_visit_update(self, episode_trajectory_part, G, S, A):
        if not (S, A) in [(s[0], s[1]) for s in episode_trajectory_part]:
            self._returns[S, A] += 1 
            #Update action-value table
            self._Q[S, A] += (1 / self._returns[S, A]) * (G - self._Q[S, A])
            #Update Policy
            self._update_policy(S)

    def _every_visit_update(self, G, S, A):
            self._returns[S, A] += 1
            #Update action-value table
            self._Q[S, A] += (1 / self._returns[S, A]) * (G - self._Q[S, A])
            #Update Policy
            self._update_policy(S)

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

    #Select mod flag between "test" and "train"
    def _play_episode(self, mod: str) -> List:
        self.reset()
        while not self._episode_ended:
            self.run_step(mod=mod)
        
        if mod == "train":
            return self._episode_trajectory
        return self._test_results
    
    def _update(self):
        G = 0
        for i in reversed(range(len(self._episode_trajectory))):
            S, A, R = self._episode_trajectory[i]
            G = (G * self._gamma) + R #Update expected return
            if self._visit_update == "first":
                self._first_visit_update(self._episode_trajectory[0:i], G, S, A)
            elif self._visit_update == "every":
                self._every_visit_update(G, S, A)
    
    def reset(self):
        self._episode_ended = False
        self._S = self._env.reset_env()
        self._episode_trajectory = []
        self._test_results = defaultdict(float)

    def run_step(self, *args, **kwargs):
        #Select action according to policy distribution probability
        A = np.random.choice(range(self._env.actions_size()), p=self._policy[self._S])
        n_S, R, self._episode_ended, test_info = self._env.run_step(A, kwargs['mod'])
        self._episode_trajectory.append((self._S, A, R))
        if kwargs['mod'] == "test":
            for test in test_info.keys():
                self._test_results[test] += test_info[test]

        if self._episode_ended:
            self._update()

        self._S = n_S
