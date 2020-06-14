from random import shuffle, randint
from typing import Dict, List
from tqdm import tqdm

import pandas as pd

from ..agents import Agent
from ..environments import Environment


class SingleTrainSession:

    __slots__ = ['_env', '_agents', '_step_start']

    _env: Environment
    _agents: List[Agent]
    _step_start: int

    def __init__(self, env: Environment, step_start: int = 1):
        self._env: Environment = env
        self._agents: Dict[int, Agent] = {}
        self._step_start = step_start-1

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
        for key in self._agents:
            step = 0
            done = False
            agent = self._agents[key]
            
            agent.reset(self._env, id=key)
            while not done:
                S, R, done, info = agent.run_step(self._env, id=key, mode='train', t=step)
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
            all_agents = list(self._agents.keys())
            agents = self._random_start(all_agents)
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
                if self._step_start >= 0:
                    agents.extend(self._random_start(all_agents))
                    self._step_start -= 1
            self.reset_env()
        return pd.DataFrame(out)

    def reset_env(self):
        for key, agent in self._agents.items():
            agent.reset(self._env, id=key)

    def _random_start(self, agent_list: List) -> List:
        all_agents = agent_list[::]
        agents = []
        for agent in all_agents:
            if randint(0, self._step_start) == 0:
                agents.append(agent)
                agent_list.remove(agent)
        return agents