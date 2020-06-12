from random import shuffle
from typing import Dict, List

import pandas as pd

from tqdm import tqdm

from ..agents import Agent
from ..environments import Environment


class Session:

    __slots__ = ['_env', '_agents']

    _env: Environment
    _agents: List[Agent]

    def __init__(self, env: Environment):
        self._env: Environment = env
        self._agents: Dict[int, Agent] = {}

    def add_agent(self, agent: Agent):
        self._agents[id(agent)] = agent

    def run(self, episodes: int, test_offset: int, test_samples: int, render: bool = False):
        self.reset_env()
        out = []
        for episode in tqdm(range(1, episodes + 1)):
            self._run_train()
            if episode % test_offset == 0:
                out.append(
                    self._run_test(episode // test_offset, test_samples, render)
                )
        return pd.concat(out)

    def _run_train(self):
        self.reset_env()
        step = 0
        agents = list(self._agents.keys())
        while len(agents) > 0:
            shuffle(agents)
            for agent in agents[::]:
                S, R, done, info = self._agents[agent].\
                    run_step(self._env, id=agent, mode='train', t=step)
                if done:
                    agents.remove(agent)
            step += 1
        
    def _run_test(self, test: int, test_samples: int, render: bool = False):
        self.reset_env()
        if render:
            self._env.render()
        out = []
        labels = {
            key: 'ID: ' + str(key) + ', Params: ' + str(value)
            for key, value in self._agents.items()
        }
        for sample in range(test_samples):
            step = 0
            agents = list(self._agents.keys())
            while len(agents) > 0:
                shuffle(agents)
                for agent in agents[::]:
                    S, R, done, info = self._agents[agent].\
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
            self.reset_env()
        return pd.DataFrame(out)

    def reset_env(self):
        for key, agent in self._agents.items():
            agent.reset(self._env, id=key)
