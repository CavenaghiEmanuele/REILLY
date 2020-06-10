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
        if render:
            self._env.render()
        results = []
        for episode in tqdm(range(episodes)):
            self._run_train()
            if episode % test_offset == 0:
                results.append(self._run_test(test_samples))
        return self._format_results(results)

    def _run_train(self):
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
        self.reset_env()

    def _run_test(self, test_samples: int):
        outs = []
        for sample in range(test_samples):
            out = []
            step = 0
            agents = list(self._agents.keys())
            while len(agents) > 0:
                shuffle(agents)
                for agent in agents[::]:
                    S, R, done, info = self._agents[agent].\
                        run_step(self._env, id=agent, mode='test', t=step)
                    if done:
                        agents.remove(agent)
                    out.append((agent, S, R, done, info))
                outs.append(out)
                step += 1
            self.reset_env()
        return outs

    def _format_results(self, data):
        labels = {
            key: 'ID: ' + str(key) + ', Params: ' + str(value)
            for key, value in self._agents.items()
        }
        out = [
            {'test': i, 'sample': j, 'step': k, 'agent': labels[agent], **info}
            for i, test in enumerate(data)
            for j, sample in enumerate(test)
            for k, (agent, state, reward, done, info) in enumerate(sample)
        ]
        out = pd.DataFrame(out)
        return out

    def reset_env(self):
        for key, agent in self._agents.items():
            agent.reset(self._env, id=key)
