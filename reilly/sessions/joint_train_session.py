from random import shuffle, randint
from typing import Dict, List
from tqdm import tqdm

import pandas as pd

from .session import Session
from ..agents import Agent
from ..environments import Environment


class JointTrainSession(Session):

    def _run_train(self) -> None:
        self._reset_env()
        step = 0
        agents = self._random_start(step)
                
        while len(agents) > 0:
            shuffle(agents)
            for agent in agents[::]:
                next_state, reward, done, info = self._agents[agent].\
                    run_step(self._env, id=agent, mode='train', t=step)
                if done:
                    agents.remove(agent)
            step += 1
            if self._start_step - step >= 0:
                agents = list(set(agents) | set(self._random_start(step)))
