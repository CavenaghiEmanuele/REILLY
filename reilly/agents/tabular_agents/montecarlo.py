import numpy as np
from typing import List

from ..tabular_agents import TabularAgent
from ...structures import ActionValue, Policy


class MonteCarlo(TabularAgent):

    __slots__ = ["_visit_update", "_policy_method", "_returns", '_episode_trajectory']

    def __init__(self,
                 states: int,
                 actions: int,
                 epsilon: float,
                 gamma: float,
                 epsilon_decay: float = 1,
                 visit_update: str = "first",
                 policy_method: str = "on-policy"):
        # Basic attribute
        self._Q = ActionValue(states, actions)
        self._policy = Policy(states, actions)
        self._returns = np.zeros(shape=(states, actions))
        self._epsilon = epsilon
        self._gamma = gamma
        self._e_decay = epsilon_decay
        self._actions = actions

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
            self._policy_update(S, self._policy, self._Q)

    def _every_visit_update(self, G: float, S: int, A: int) -> None:
        self._returns[S, A] += 1
        # Update action-value table
        self._Q[S, A] += (1 / self._returns[S, A]) * (G - self._Q[S, A])
        # Update Policy
        self._policy_update(S, self._policy, self._Q)

    def reset(self, init_state: int, *args, **kwargs) -> None:
        self._S = init_state
        self._A = self._select_action(self._policy[init_state])
        self._episode_trajectory = []

    def update(self, n_S: int, R: float, done: bool, *args, **kwargs) -> None:
        # Select action according to policy distribution probability
        A = self._select_action(self._policy[n_S])

        self._episode_trajectory.append((self._S, A, R))
        if kwargs['training'] and done:
            self._epsilon *= self._e_decay
            G = 0
            for i in reversed(range(len(self._episode_trajectory))):
                S, A, R = self._episode_trajectory[i]
                G = (G * self._gamma) + R  # Update expected return
                if self._visit_update == "first":
                    self._first_visit_update(
                        self._episode_trajectory[0:i], G, S, A)
                elif self._visit_update == "every":
                    self._every_visit_update(G, S, A)

        self._S = n_S
        self._A = A
