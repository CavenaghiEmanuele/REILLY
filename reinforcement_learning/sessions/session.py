from random import shuffle, randint
from typing import Dict, List
from tqdm import tqdm
from abc import ABC, abstractmethod

import pandas as pd

from ..agents import Agent
from ..environments import Environment


class Session(ABC):

    __slots__ = ['_env', '_agents', '_start_step']

    _env: Environment
    _agents: List[Agent]
    _start_step: int

    def __init__(self, env: Environment, start_step: int = 1):
        self._env: Environment = env
        self._agents: Dict[int, Agent] = {}
        self._start_step = start_step - 1

    def add_agent(self, agent: Agent):
        self._agents[id(agent)] = agent

    def run(self, episodes: int, test_offset: int, test_samples: int, render: bool = False) -> pd.DataFrame:
        self._reset_env()
        out = []
        for episode in tqdm(range(1, episodes + 1)):
            self._run_train()
            if episode % test_offset == 0:
                out.append(
                    self._run_test(episode // test_offset,
                                   test_samples, render)
                )
        return pd.concat(out)

    @abstractmethod
    def _run_train(self) -> None:
        pass

    def _run_test(self, test: int, test_samples: int, render: bool = False) -> pd.DataFrame:
        self._reset_env()
        if render:
            self._env.render()
        out = []
        labels = {
            key: 'ID: ' + str(key) + ', Params: ' + str(value)
            for key, value in self._agents.items()
        }
        for sample in range(test_samples):
            step = 0
            agents = self._random_start()
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
                        'agent': labels[agent],
                        **info
                    })
                step += 1
                if self._start_step > 0:
                    self._start_step -= 1
                    agents = list(set(agents) + set(self._random_start()))
            self._reset_env()
        return pd.DataFrame(out)

    def _reset_env(self) -> None:
        for key, agent in self._agents.items():
            agent.reset(self._env, id=key)

    def _random_start(self) -> List:
        return [
            agent
            for agent in self._agents.keys()
            if randint(0, self._start_step) == 0
        ]
