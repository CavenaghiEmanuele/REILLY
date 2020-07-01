from random import shuffle, randint
from typing import Dict, List
from tqdm import tqdm
from abc import ABC, abstractmethod

import pandas as pd

from ..agents import Agent
from ..environments import Environment


class Session(ABC):

    __slots__ = ['_env', '_agents', '_labels', '_start_step']

    _env: Environment
    _agents: Dict[int, Agent]
    _labels: Dict[int, str]
    _start_step: int

    def __init__(self, env: Environment, start_step: int = 1):
        self._env = env
        self._agents = {}
        self._labels = {}
        self._start_step = start_step - 1

    def add_agent(self, agent: Agent):
        key = id(agent)
        self._agents[key] = agent
        self._labels[key] = 'ID: ' + str(key) + ', Params: ' + str(agent)

    def run(self, episodes: int, test_offset: int, test_samples: int, render: bool = False, heatmap: bool = False) -> pd.DataFrame:
        self._reset_env()
        out = []
        for episode in tqdm(range(1, episodes + 1)):
            self._run_train()
            if episode % test_offset == 0:
                out.append(
                    self._run_test(episode // test_offset,
                                   test_samples, render, heatmap)
                )
        return pd.concat(out)

    @abstractmethod
    def _run_train(self) -> None:
        pass

    def _run_test(self, test: int, test_samples: int, render: bool = False, heatmap: bool = False) -> pd.DataFrame:
        self._reset_env()
        if render:
            self._env.render()
        if heatmap:
            self._env.heatmap()
        out = []
        for sample in range(test_samples):
            step = 0
            agents = self._random_start(step)
            while len(agents) > 0:
                shuffle(agents)
                for agent in agents[::]:
                    next_state, reward, done, info = self._agents[agent].\
                        run_step(self._env, id=agent, mode='test', t=step)
                    if done:
                        agents.remove(agent)
                    out.append({
                        'test': test,
                        'sample': sample,
                        'step': step,
                        'agent': self._labels[agent],
                        **info
                    })
                step += 1
                if self._start_step - step >= 0:
                    agents = list(set(agents) | set(self._random_start(step)))
            self._reset_env()
        return pd.DataFrame(out)

    def _reset_env(self) -> None:
        for key, agent in self._agents.items():
            agent.reset(self._env, id=key)

    def _random_start(self, step) -> List:
        return [
            agent
            for agent in self._agents.keys()
            if randint(0, self._start_step - step) == 0
        ]