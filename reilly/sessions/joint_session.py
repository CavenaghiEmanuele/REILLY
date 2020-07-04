from random import shuffle, randint
from typing import Dict, List
from tqdm import tqdm
from abc import ABC, abstractmethod

import pandas as pd

from ..agents import Agent
from ..environments import Environment


class JointSession(ABC):

    __slots__ = ['_env', '_agents', '_labels', '_start_step']

    _env: Environment
    _agents: Dict[int, Agent]
    _labels: Dict[int, str]
    _start_step: int
    _joint_train: bool

    def __init__(self, env: Environment, agents: List[Agent], start_step: int = 1, joint_train=False):
        self._env = env
        self._agents = {id(agent): agent for agent in agents}
        self._labels = {
            id(agent): "ID: {}, Params: {}".format(id(agent), agent)
            for agent in agents
        }
        self._start_step = start_step - 1
        self._joint_train = joint_train

    def add_agent(self, agent: Agent):
        key = id(agent)
        self._agents[key] = agent
        self._labels[key] = "ID: {}, Params: {}".format(key, agent)

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

    def _run_train(self) -> None:
        if not self._joint_train:
            self._run_train_single()
        if self._joint_train:
            self._run_train_joint()

    def _run_train_single(self) -> None:
        self._reset_env()
        step = 0
        agents = self._random_start(step)

        while len(agents) > 0:
            shuffle(agents)
            for agent in agents[::]:
                action = self._agents[agent].get_action()
                next_state, reward, done, _ = self._env.run_step(
                    action,
                    id=agent,
                    mode='test',
                    t=step
                )
                self._agents[agent].update(
                    next_state,
                    reward,
                    done,
                    training=True,
                    t=step
                )
                if done:
                    agents.remove(agent)
            step += 1
            if self._start_step - step >= 0:
                agents = list(set(agents) | set(self._random_start(step)))

    def _run_train_joint(self):
        for key, agent in self._agents.items():
            step = 0
            done = False

            agent.reset(self._env, id=key)
            while not done:
                action = agent.get_action()
                next_state, reward, done, _ = self._env.run_step(
                    action,
                    id=key,
                    mode='test',
                    t=step
                )
                agent.update(
                    next_state,
                    reward,
                    done,
                    training=True,
                    t=step
                )
                step += 1

    def _run_test(self, test: int, test_samples: int, render: bool = False) -> pd.DataFrame:
        self._reset_env()
        if render:
            self._env.render()
        out = []
        for sample in range(test_samples):
            step = 0
            agents = self._random_start(step)
            while len(agents) > 0:
                shuffle(agents)
                for agent in agents[::]:
                    action = self._agents[agent].get_action()
                    next_state, reward, done, info = self._env.run_step(
                        action,
                        id=agent,
                        mode='test',
                        t=step
                    )
                    self._agents[agent].update(
                        next_state,
                        reward,
                        done,
                        training=True,
                        t=step
                    )
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
