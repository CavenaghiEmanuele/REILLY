import numpy as np
from typing import List, Dict

from ....structures import ActionValue, Policy
from .n_step import NStep


class NStepExpectedSarsaAgent(NStep, object):

    __slots__ = ['_states', '_actions', '_rewards', 'T']

    def __repr__(self):
        return "n-step Expected Sarsa: " + "alpha=" + str(self._alpha) + ", gamma=" + str(self._gamma) + ", epsilon=" + str(self._epsilon) + ", n-step=" + str(self._n_step)

    def reset(self, env):
        self._episode_ended = False
        self._states = [env.reset()]
        self._actions = [np.random.choice(
            range(env.actions_size), p=self._policy[self._states[0]])]
        self._rewards = [0.0]
        self.T = float('inf')

    def run_step(self, env, *args, **kwargs):
        t = kwargs['t']
        if t < self.T:
            n_S, R, self._episode_ended, info = env.run_step(
                self._actions[t], **kwargs)
            self._states.append(n_S)
            self._rewards.append(R)

            if self._episode_ended == True:
                self.T = t + 1
            else:
                self._actions.append(np.random.choice(
                    range(env.actions_size), p=self._policy[n_S]))

        pi = t - self._n_step + 1
        if pi >= 0:
            G = 0
            # Don't add +1 in upper bound of sum because the last element is
            # a branch over all action possibilities weighted by their probability
            for i in range(pi + 1, min(self.T, pi + self._n_step)):
                G += np.power(self._gamma, i - pi - 1) * self._rewards[i]

            G += self._compute_expected_value(
                state=self._states[pi + self._n_step])

            if pi + self._n_step < self.T:
                G += np.power(self._gamma, self._n_step) * \
                    self._Q[self._states[pi + self._n_step]
                            ][self._actions[pi + self._n_step]]

            self._Q[self._states[pi], self._actions[pi]] += self._alpha * \
                (G - self._Q[self._states[pi], self._actions[pi]])
            self._update_policy(self._states[pi])

        return (n_S, R, self._episode_ended, info)

    def _compute_expected_value(self, state) -> float:
        expected_value = 0
        for action in range(len(self._Q[state])):
            expected_value += self._policy[state,
                                           action] * self._Q[state, action]
        return expected_value
