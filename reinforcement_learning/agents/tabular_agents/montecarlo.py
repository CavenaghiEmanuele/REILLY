import numpy as np
from typing import List, Tuple
from collections import defaultdict

from ..agent import Agent
from ...structures import ActionValue, Policy
from ...environments import Environment


class MonteCarloAgent(Agent):

    __slots__ = ["_visit_update", "_policy_method", "_returns", '_episode_trajectory']

    def __init__(self,
                 states_size: int,
                 actions_size: int,
                 epsilon: float,
                 gamma: float,
                 epsilon_decay: float = 1,
                 visit_update: str = "first",
                 policy_method: str = "on-policy"):
        # Basic attribute
        self._Q = ActionValue(states_size, actions_size)
        self._policy = Policy(states_size, actions_size)
        self._returns = np.zeros(shape=(states_size, actions_size))
        self._epsilon = epsilon
        self._gamma = gamma
        self._e_decay = epsilon_decay

        # Flags
        self._visit_update = visit_update
        self._policy_method = policy_method

    def __repr__(self):
        return  "MonteCarlo: " +\
                "gamma=" + str(self._gamma) +\
                ", epsilon=" + str(self._epsilon) +\
                ", e-decay=" + str(self._e_decay)

    def _first_visit_update(self,
                            episode_trajectory_part:List,
                            G:float,
                            S:int,
                            A:int) -> None:
        if not (S, A) in [(s[0], s[1]) for s in episode_trajectory_part]:
            self._returns[S, A] += 1
            # Update action-value table
            self._Q[S, A] += (1 / self._returns[S, A]) * (G - self._Q[S, A])
            # Update Policy
            self._update_policy(S)

    def _every_visit_update(self, G:float, S:int, A:int) -> None:
        self._returns[S, A] += 1
        # Update action-value table
        self._Q[S, A] += (1 / self._returns[S, A]) * (G - self._Q[S, A])
        # Update Policy
        self._update_policy(S)

    def _update_policy(self, S:int) -> None:
        # Avoid choosing always the first move in case policy has the same value
        indices = [i for i, x in enumerate(self._Q[S]) if x == max(self._Q[S])]
        A_star = np.random.choice(indices)

        n_actions = self._policy.get_n_actions(S)
        for A in range(n_actions):
            if A == A_star:
                self._policy[S, A] = 1 - self._epsilon + \
                    (self._epsilon / n_actions)
            else:
                self._policy[S, A] = self._epsilon / n_actions

    def _update(self) -> None:
        G = 0
        for i in reversed(range(len(self._episode_trajectory))):
            S, A, R = self._episode_trajectory[i]
            G = (G * self._gamma) + R  # Update expected return
            if self._visit_update == "first":
                self._first_visit_update(
                    self._episode_trajectory[0:i], G, S, A)
            elif self._visit_update == "every":
                self._every_visit_update(G, S, A)

    def reset(self, env:Environment, *args, **kwargs) -> None:
        self._episode_ended = False
        self._S = env.reset(*args, **kwargs)
        self._episode_trajectory = []

    def run_step(self, env:Environment, *args, **kwargs) -> Tuple:
        # Select action according to policy distribution probability
        A = np.random.choice(range(env.actions_size),
                             p=self._policy[self._S])
        n_S, R, self._episode_ended, info = env.run_step(A, **kwargs)
        self._episode_trajectory.append((self._S, A, R))
        if not kwargs['mode'] == "test" and self._episode_ended:
            self._update()

        self._S = n_S
        if self._episode_ended:
            self._epsilon *= self._e_decay
        return (n_S, R, self._episode_ended, info)
