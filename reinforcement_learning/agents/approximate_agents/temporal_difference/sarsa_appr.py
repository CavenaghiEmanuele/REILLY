import numpy as np

from .temporal_difference_appr import TemporalDiffernceAppr


class SarsaApproximateAgent(TemporalDiffernceAppr, object):

    __slots__ = ['_A']

    def __repr__(self):
        return "Sarsa Appr: " + "alpha=" + str(self._alpha) + \
            ", gamma=" + str(self._gamma) + \
            ", epsilon=" + str(self._epsilon)

    def reset(self, env):
        self._episode_ended = False
        self._S = env.reset_env()
        self._A = np.random.choice(
            range(env.actions_size()), p=self._e_greedy_policy(env, self._S))

    def run_step(self, env, *args, **kwargs):

        n_S, R, self._episode_ended, info = env.run_step(self._A, **kwargs)

        if self._episode_ended:
            self._Q_estimator.update(self._S, self._A, R)
            return (n_S, R, self._episode_ended, info)

        n_A = np.random.choice(range(env.actions_size()),
                               p=self._e_greedy_policy(env, n_S))

        G = R + (self._gamma * self._Q_estimator.predict(n_S, n_A))
        self._Q_estimator.update(self._S, self._A, G)

        self._S = n_S
        self._A = n_A
        return (n_S, R, self._episode_ended, info)
