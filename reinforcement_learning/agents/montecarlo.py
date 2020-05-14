import numpy as np
from typing import List, Dict
from collections import defaultdict
from tqdm import tqdm

from ..structures import ActionValue, Policy
from ..environments.environment import Environment


class MonteCarloAgent(object):

    __slots__ = ["_epsilon", "_gamma", "_visit_update", "_policy_method", "_returns", "_Q", "_policy", "_env"]

    def __init__(self, epsilon, gamma, environment, visit_update="first", policy_method="on-policy"):
        #Basic attribute
        self._env = environment
        n_states = self._env.get_state_number()
        n_actions = self._env.get_action_number()
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
            self._MonteCarlo_control()
            if (n_episode % test_step) == 0:
                test_info = self.test(n_tests)
                [test_results[test].append(test_info[test]) for test in test_info.keys()]
        return test_results

    def train(self, n_episodes: int) -> None:
        for _ in tqdm(range(n_episodes)):
            self._MonteCarlo_control()
    
    #Repeat test n_tests time to avoid outliers results
    def test(self, n_tests: int):
        test_results = defaultdict(list)
        for _ in range(n_tests):
            test_info = self._play_episode(mod="test")
            [test_results[test].append(test_info[test]) for test in test_info.keys()]
        return {test : np.average(test_results[test]) for test in test_results.keys()}
   
    def _MonteCarlo_control(self) -> None:
        episode_trajectory = self._play_episode(mod="train") #Play an entire episode
        G = 0
        for i in reversed(range(len(episode_trajectory))):
            s_t, a_t, r_t = episode_trajectory[i]
            G = (G * self._gamma) + r_t #Update expected return
            if self._visit_update == "first":
                self._first_visit_update(episode_trajectory[0:i], G, s_t, a_t)
            elif self._visit_update == "every":
                self._every_visit_update(G, s_t, a_t)

    def _first_visit_update(self, episode_trajectory_part, G, s_t, a_t):
        if not (s_t, a_t) in [(s[0], s[1]) for s in episode_trajectory_part]:
            self._returns[s_t, a_t] += 1 
            #Update action-value table
            self._Q[s_t, a_t] += (1 / self._returns[s_t, a_t]) * (G - self._Q[s_t, a_t])
            #Update Policy
            self._update_policy(s_t)

    def _every_visit_update(self, G, s_t, a_t):
            self._returns[s_t, a_t] += 1
            #Update action-value table
            self._Q[s_t, a_t] += (1 / self._returns[s_t, a_t]) * (G - self._Q[s_t, a_t])
            #Update Policy
            self._update_policy(s_t)

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

    #Select mod flag between "test" and "train"
    def _play_episode(self, mod: str) -> List:
        state = self._env.reset_env()
        episode_ended = False
        episode_trajectory = []
        test_results = defaultdict(float)

        while not episode_ended:
            #Select action according to policy distribution probability
            action = np.random.choice(range(self._env.get_action_number()), p=self._policy[state])
            if mod == "train":
                next_state, reward, episode_ended, _ = self._env.run_step(action, mod)
                episode_trajectory.append((state, action, reward))
            elif mod == "test":
                next_state, reward, episode_ended, test_info = self._env.run_step(action, mod)
                for test in test_info.keys():
                    test_results[test] += test_info[test]    
            state = next_state
        
        if mod == "train":
            return episode_trajectory
        elif mod == "test":
            return test_results
