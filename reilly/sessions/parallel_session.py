from typing import Dict
from itertools import product
from multiprocessing import Pool, cpu_count

import pandas as pd

from .. import agents as _agents
from .. import environments as _envs
from .session import Session


def runner(envs, agents, configs):
    env = getattr(_envs, envs[0])(**envs[1])
    agent = getattr(_agents, agents[0])(**{
        'states': env.states,
        'actions': env.actions,
        **agents[1]
    })
    session = Session(env, agent)
    return session.run(**configs)


class ParallelSession():

    def __init__(self, envs: Dict, agents: Dict, *args, **kwargs):
        self._envs = self._configs_to_params(envs)
        self._agents = self._configs_to_params(agents)

    def _configs_to_params(self, configs: Dict):
        return [
            (key, dict(zip(config.keys(), params)))
            for key, config in configs.items()
            for params in product(*[
                val if isinstance(val, list) else [val]
                for val in config.values()
            ])
        ]

    def run(self, episodes: int, test_offset: int, test_samples: int, *args, **kwargs):
        params = list(product(
            self._envs,
            self._agents,
            [{
                'episodes': episodes,
                'test_offset': test_offset,
                'test_samples': test_samples,
            }]
        ))

        pool = Pool(kwargs.get('cpu_count', cpu_count()))
        data = pool.starmap(runner, params)
        pool.close()
        pool.join()

        return pd.concat(data)
