from random import shuffle, randint
from typing import Dict, List
from tqdm import tqdm

import pandas as pd

from .session import Session
from ..agents import Agent
from ..environments import Environment


class JoinTrainSession(Session):

    def _run_train(self) -> None:
        self.reset_env()
        step = 0
        all_agents = list(self._agents.keys())
        agents = self._random_start(all_agents)
                
        while len(agents) > 0:
            shuffle(agents)
            for agent in agents[::]:
                S, R, done, info = self._agents[agent].\
                    run_step(self._env, id=agent, mode='train', t=step)
                if done:
                    agents.remove(agent)
            step += 1
            if self._step_start > 0:
                self._step_start -= 1
                agents.extend(self._random_start(all_agents))