import numpy as np

from .n_step import NStep


class NStepSarsa(NStep, object):

    def __repr__(self):
        return "n-step Sarsa: " + "alpha=" + str(self._alpha) + \
            ", gamma=" + str(self._gamma) + \
            ", epsilon=" + str(self._epsilon) + \
            ", n-step=" + str(self._n_step) + \
            ", e-decay=" + str(self._e_decay)

    def update(self, n_S: int, R: float, done: bool, *args, **kwargs) -> None:
        t = kwargs['t']
        if t < self.T:
            self._states.append(n_S)
            self._rewards.append(R)

            if done: 
                self.T = t + 1
                self._epsilon *= self._e_decay
            else:
                self._action_list.append(self._select_action(self._policy[n_S]))

        if kwargs['training']:
            pi = t - self._n_step + 1
            if pi >= 0:
                G = 0
                for i in range(pi + 1, min(self.T, pi + self._n_step)):
                    G += np.power(self._gamma, i - pi - 1) * self._rewards[i]

                if pi + self._n_step < self.T:
                    G += np.power(self._gamma, self._n_step) * \
                        self._Q[self._states[pi + self._n_step]][self._action_list[pi + self._n_step]]
                        
                self._Q[self._states[pi], self._action_list[pi]] += self._alpha * \
                    (G - self._Q[self._states[pi], self._action_list[pi]])
                self._policy_update(self._states[pi], self._policy, self._Q)
