import numpy as np
from typing import List, Dict
from collections import defaultdict

from .agent import Agent
from ..structures import ActionValue, Policy


class MonteCarloAgent(Agent):

    __slots__ = ["_visit_update", "_policy_method",
                 "_returns", '_episode_trajectory', '_test_results']

    def __init__(self, states_size, actions_size, epsilon, gamma, visit_update="first", policy_method="on-policy"):
        # Basic attribute
        self._Q = ActionValue(states_size, actions_size)
        self._policy = Policy(states_size, actions_size)
        self._returns = np.zeros(shape=(states_size, actions_size))
        self._epsilon = epsilon
        self._gamma = gamma

        # Flags
        self._visit_update = visit_update
        self._policy_method = policy_method

    def __repr__(self):
        return "MonteCarlo: " + "gamma=" + str(self._gamma) + ", epsilon=" + str(self._epsilon)

    def _first_visit_update(self, episode_trajectory_part, G, S, A):
        if not (S, A) in [(s[0], s[1]) for s in episode_trajectory_part]:
            self._returns[S, A] += 1
            # Update action-value table
            self._Q[S, A] += (1 / self._returns[S, A]) * (G - self._Q[S, A])
            # Update Policy
            self._update_policy(S)

    def _every_visit_update(self, G, S, A):
        self._returns[S, A] += 1
        # Update action-value table
        self._Q[S, A] += (1 / self._returns[S, A]) * (G - self._Q[S, A])
        # Update Policy
        self._update_policy(S)

    def _update_policy(self, S) -> None:
        # Avoid choosing always the first move in case policy has the same value
        indices = [i for i, x in enumerate(self._Q[S]) if x == max(self._Q[S])]
        A_star = np.random.choice(indices)

        n_actions = self._policy.get_n_actions(S)
        for A in range(n_actions):
            if A == A_star:
                self._policy[S, A] = 1 - self._epsilon + \
                    (self._epsilon/n_actions)
            else:
                self._policy[S, A] = self._epsilon/n_actions

    def _update(self):
        G = 0
        for i in reversed(range(len(self._episode_trajectory))):
            S, A, R = self._episode_trajectory[i]
            G = (G * self._gamma) + R  # Update expected return
            if self._visit_update == "first":
                self._first_visit_update(
                    self._episode_trajectory[0:i], G, S, A)
            elif self._visit_update == "every":
                self._every_visit_update(G, S, A)

    def reset(self, env):
        self._episode_ended = False
        self._S = env.reset()
        self._episode_trajectory = []
        self._test_results = defaultdict(float)

    def run_step(self, env, *args, **kwargs):
        # Select action according to policy distribution probability
        A = np.random.choice(range(env.actions_size()),
                             p=self._policy[self._S])
        n_S, R, self._episode_ended, info = env.run_step(A, **kwargs)
        self._episode_trajectory.append((self._S, A, R))
        if kwargs['mode'] == "test":
            for test in info.keys():
                self._test_results[test] += info[test]

        if self._episode_ended:
            self._update()

        self._S = n_S

        return (n_S, R, self._episode_ended, info)
