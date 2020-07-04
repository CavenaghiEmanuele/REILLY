from typing import Dict, List
from tqdm import trange

import pandas as pd

from ..agents import Agent
from ..environments import Environment


class Session(object):

    __slots__ = ['_env', '_agent', '_label', '_position']

    _env: Environment
    _agent: Agent
    _label: str
    _position: int

    def __init__(self, env: Environment, agent: Agent, position: int = 0):
        self._env = env
        self._agent = agent
        self._label = "ID: {}, Params: {}".format(id(agent), agent)
        self._position = position

    def run(self, episodes: int, test_offset: int, test_samples: int, render: bool = False) -> pd.DataFrame:
        out = []
        self._reset_env()
        for episode in trange(1, episodes + 1, position=self._position, leave=True):
            self._run_train()
            if episode % test_offset == 0:
                out.append(
                    self._run_test(episode // test_offset,
                                   test_samples, render)
                )
        return pd.concat(out)

    def _run_train(self) -> None:
        step = 0
        done = False
        while not done:
            action = self._agent.get_action()
            next_state, reward, done, _ = self._env.run_step(
                action,
                id=id(self._agent),
                mode='test',
                t=step
            )
            self._agent.update(
                next_state,
                reward,
                done,
                training=True,
                t=step
            )
            step += 1
        self._reset_env()

    def _run_test(self, test: int, test_samples: int, render: bool = False) -> pd.DataFrame:
        self._reset_env()
        if render:
            self._env.render()
        out = []
        for sample in range(test_samples):
            step = 0
            done = False
            while not done:
                action = self._agent.get_action()
                next_state, reward, done, info = self._env.run_step(
                    action,
                    id=id(self._agent),
                    mode='test',
                    t=step
                )
                self._agent.update(
                    next_state,
                    reward,
                    done,
                    training=False,
                    t=step
                )
                out.append({
                    'test': test,
                    'sample': sample,
                    'step': step,
                    'agent': self._label,
                    **info
                })
                step += 1
            self._reset_env()
        return pd.DataFrame(out)

    def _reset_env(self) -> None:
        init_state = self._env.reset(id=id(self._agent))
        self._agent.reset(init_state)
