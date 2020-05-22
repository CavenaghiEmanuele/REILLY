from random import shuffle
from typing import Dict, List

from tqdm import tqdm

from ..agents import Agent
from ..environments import Environment


class Session:

    def __init__(self, env: Environment):
        self._env: Environment = env
        self._agents: Dict[int, Agent] = {}

    def add_agent(self, agent: Agent):
        self._agents[id(agent)] = agent

    def run(self, episodes: int, test_offset: int, test_samples: int):
        results = []
        for episode in tqdm(range(episodes)):
            self._run_train()
            if episode % test_offset == 0:
                result = self._run_test(test_samples)
                results.append(result)
        return results

    def _run_train(self):
        step = 0
        agents = list(self._agents.keys())
        while len(agents) > 0:
            shuffle(agents)
            for agent in agents:
                S, R, done, info = self._agents[agent].\
                    run_step(self._env, t=step, mode='train')
            if done:
                agents.remove(agent)
            step += 1
        self.reset_env()
    
    def _run_test(self, test_samples: int):
        tests = self._env.get_env_tests()
        results = {
            agent: {test: [] for test in tests}
            for agent in self._agents.keys()
        }
        for sample in range(test_samples):
            step = 0
            result = {
                agent: {test: 0 for test in tests}
                for agent in self._agents.keys()
            }
            agents = list(self._agents.keys())
            while len(agents) > 0:
                shuffle(agents)
                for agent in agents:
                    S, R, done, info = self._agents[agent].\
                        run_step(self._env, t=step, mode='test')
                    for i, value in info.items():
                        result[agent][i] += value
                if done:
                    agents.remove(agent)
                step += 1
            self.reset_env()
            for key, res in result.items():
                for k, v in res.items():
                    results[key][k].append(v)
        for key, res in results.items():
            for k in res.keys():
                results[key][k] = sum(results[key][k]) / len(results[key][k])
        return results

    def reset_env(self):
        for agent in self._agents.values():
            agent.reset(self._env)
