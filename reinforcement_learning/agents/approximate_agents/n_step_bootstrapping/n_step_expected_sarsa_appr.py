import numpy as np

from .n_step_appr import NStepAppr


class NStepExpectedSarsaApproximateAgent(NStepAppr, object):

    __slots__ = ['_states', '_actions', '_rewards', 'T']

    def __repr__(self):
        return "n-step Expected Sarsa Appr: " + "alpha=" + str(self._alpha) + \
            ", gamma=" + str(self._gamma) + \
            ", epsilon=" + str(self._epsilon) + \
            ", n-step=" + str(self._n_step)

    def _compute_expected_value(self, state, n_action) -> float:
        expected_value = 0
        for action in range(n_action):
            expected_value += self._e_greedy_policy(
                state, n_action)[action] * self._Q_estimator.predict(state, action)
        return expected_value

    def reset(self, env):
        self._episode_ended = False
        self._states = [env.reset()]
        self._actions = [np.random.choice(
            range(env.actions_size), p=self._e_greedy_policy(self._states[0], env.actions_size))]
        self._rewards = [0.0]
        self.T = float('inf')

    def run_step(self, env, *args, **kwargs):
        t = kwargs['t']
        n_S = R = 0
        info = {}
        if t < self.T:
            n_S, R, self._episode_ended, info = env.run_step(
                self._actions[t], **kwargs)
            self._states.append(n_S)
            self._rewards.append(R)

            if self._episode_ended == True:
                self.T = t + 1
            else:
                self._actions.append(np.random.choice(
                    range(env.actions_size), p=self._e_greedy_policy(n_S, env.actions_size)))

        pi = t - self._n_step + 1
        if pi >= 0:
            G = 0

            for i in range(pi + 1, min(self.T, pi + self._n_step) + 1):
                G += np.power(self._gamma, i - pi - 1) * self._rewards[i]

            if pi + self._n_step < self.T:
                G += np.power(self._gamma, self._n_step) * self._compute_expected_value(
                    self._states[pi + self._n_step], env.actions_size)


            self._Q_estimator.update(self._states[pi], self._actions[pi], G)

        return (n_S, R, self._episode_ended, info)
