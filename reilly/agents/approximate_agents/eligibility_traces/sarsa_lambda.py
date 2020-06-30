import numpy as np
from typing import List, Tuple

from ....agents import Agent
from ....environments import Environment
from ..q_estimator import QEstimator


class SarsaLambdaAgent(Agent, object):

    __slots__ = ['_Q_estimator', '_lambda', '_A']

    def __init__(self,
                 alpha: float,
                 epsilon: float,
                 gamma: float,
                 lambd: float,
                 feature_dims: int,
                 num_tilings: int,
                 epsilon_decay: float = 1,
                 trace_type: str = "replacing",
                 tiling_offset: List = None,
                 tiles_size: List = None):
        # self._policy  -> Approximante agents don't have policy but approximate it
        self._alpha = alpha
        self._epsilon = epsilon
        self._gamma = gamma
        self._lambda = lambd
        self._e_decay = epsilon_decay
        self._Q_estimator = QEstimator(alpha=alpha,
                                       feature_dims=feature_dims,
                                       num_tilings=num_tilings,
                                       tiling_offset=tiling_offset,
                                       tiles_size=tiles_size,
                                       have_trace=True,
                                       trace_type=trace_type
                                       )

    def __repr__(self):
        return "Sarsa Lambda: " + "alpha=" + str(self._alpha) + \
            ", gamma=" + str(self._gamma) + \
            ", epsilon=" + str(self._epsilon) + \
            ", lambda=" + str(self._lambda) + \
            ", e-decay=" + str(self._e_decay)

    def reset(self, env: Environment, *args, **kwargs) -> None:
        self._episode_ended = False
        self._S = env.reset(*args, **kwargs)
        self._A = np.random.choice(
            range(env.actions), p=self._e_greedy_policy(self._S, env.actions))
        self._Q_estimator.reset_traces()

    def run_step(self, env: Environment, *args, **kwargs) -> Tuple:
        n_S, R, self._episode_ended, info = env.run_step(self._A, **kwargs)

        if self._episode_ended and not kwargs['mode'] == "test":
            self._Q_estimator.update(self._S, self._A, R)
            return (n_S, R, self._episode_ended, info)

        n_A = np.random.choice(range(env.actions),
                               p=self._e_greedy_policy(n_S, env.actions))

        if not kwargs['mode'] == "test":
            G = R + (self._gamma * self._Q_estimator.predict(n_S, n_A))
            self._Q_estimator.update(self._S, self._A, G)
            self._Q_estimator.update_traces(self._gamma, self._lambda)

        self._S = n_S
        self._A = n_A

        if self._episode_ended:
            self._epsilon *= self._e_decay
        return (n_S, R, self._episode_ended, info)

    def _e_greedy_policy(self, state, n_actions):
        action_probs = np.zeros(n_actions)
        q_values = [self._Q_estimator.predict(state, action)
                    for action in range(n_actions)]
        indices = [i for i, x in enumerate(q_values) if x == max(q_values)]
        best_action = np.random.choice(indices)

        for action in range(n_actions):
            if action == best_action:
                action_probs[action] = 1 - self._epsilon + \
                    (self._epsilon / n_actions)
            else:
                action_probs[action] = self._epsilon / n_actions
        return action_probs
