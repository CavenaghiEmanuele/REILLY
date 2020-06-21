import numpy as np
from typing import List, Tuple

from ....structures import ActionValue, Policy
from ....environments import Environment
from .n_step import NStep


class NStepSarsaAgent(NStep, object):

    __slots__ = ['_states', '_actions', '_rewards', 'T']

    def __repr__(self):
        return "n-step Sarsa: " + "alpha=" + str(self._alpha) + \
            ", gamma=" + str(self._gamma) + \
            ", epsilon=" + str(self._epsilon) + \
            ", n-step=" + str(self._n_step) + \
            ", e-decay=" + str(self._e_decay)

    def reset(self, env: Environment, *args, **kwargs) -> None:
        self._episode_ended = False
        self._states = [env.reset(*args, **kwargs)]
        self._actions = [np.random.choice(
            range(env.actions), p=self._policy[self._states[0]])]
        self._rewards = [0.0]
        self.T = float('inf')

    def run_step(self, env: Environment, *args, **kwargs) -> Tuple:
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
                    range(env.actions), p=self._policy[n_S]))

        if not kwargs['mode'] == "test":
            pi = t - self._n_step + 1
            if pi >= 0:
                G = 0
                for i in range(pi + 1, min(self.T, pi + self._n_step)):
                    G += np.power(self._gamma, i - pi - 1) * self._rewards[i]

                if pi + self._n_step < self.T:
                    G += np.power(self._gamma, self._n_step) * \
                        self._Q[self._states[pi + self._n_step]
                                ][self._actions[pi + self._n_step]]
                        
                self._Q[self._states[pi], self._actions[pi]] += self._alpha * \
                    (G - self._Q[self._states[pi], self._actions[pi]])
                self._update_policy(self._states[pi])
        
        if self._episode_ended:
            self._epsilon *= self._e_decay
        return (n_S, R, self._episode_ended, info)
