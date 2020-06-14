from random import shuffle, randint
from typing import Dict, List
from tqdm import tqdm

import pandas as pd

from .session import Session
from ..agents import Agent
from ..environments import Environment


class SingleTrainSession(Session):

    def _run_train(self):
        for key in self._agents:
            step = 0
            done = False
            agent = self._agents[key]
            
            agent.reset(self._env, id=key)
            while not done:
                S, R, done, info = agent.run_step(self._env, id=key, mode='train', t=step)
                step += 1
        