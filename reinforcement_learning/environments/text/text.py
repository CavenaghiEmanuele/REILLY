import numpy as np

from PIL import Image
from datetime import datetime
from enum import IntEnum, auto, unique
from random import choice
from typing import Dict, List

from ..environment import Environment


@unique
class TextStates(IntEnum):
    EMPTY = 0
    SPAWN = auto()
    AGENT = auto()
    GOAL = auto()
    WALL = auto()


@unique
class TextNeighbor(IntEnum):
    NEUMANN = 4     # North, East, South, West
    MOORE = 8       # N, NE, E, SE, S, SW, W, NW


class TextEnvironment(Environment):

    __slots__ = [
        '_gui', '_env_init', '_env_exec', '_agents', '_neighbor',
        '_max_steps', '_mapper', '_rewards', '_raw_state'
    ]

    _gui: List
    _env_init: np.ndarray
    _env_exec: np.ndarray
    _agents: Dict
    _neighbor: int
    _max_steps: int
    _mapper: Dict
    _rewards: Dict
    _raw_state: bool

    def __init__(
        self,
        text: str = '##########\n          \n S        \n          \n        X \n          \n##########',
        neighbor: int = TextNeighbor.MOORE,
        max_steps: int = 50,
        rewards: Dict = {TextStates.EMPTY: -1, TextStates.WALL: -5, TextStates.GOAL: 20},
        raw_state: bool = True
    ):
        # Set default GUI to None
        self._gui = None
        # Set defualt agents to Dict
        self._agents = {}
        # Set neighborhood type
        self._neighbor = neighbor
        # Set maximun steps for agent exploration
        self._max_steps = max_steps
        # Set rewards per states
        self._rewards = rewards
        # If True return only one dimensions for states
        self._raw_state = raw_state
        # Define text mapper values
        self._mapper = {
            ' ': TextStates.EMPTY,          # Empty space
            'S': TextStates.SPAWN,          # Spawn zones
            'A': TextStates.AGENT,          # Agent pointers
            'X': TextStates.GOAL,           # Goal pointers
            '#': TextStates.WALL,           # Walls
        }
        # Remove formatting chars at beginning or end of text
        text = text.strip('\n\r\t').split('\n')
        # Check if all lines have equal length
        shape = [len(line) for line in text]
        if len(set(shape)) > 1:
            raise ValueError
        # Compute env shape
        shape = (len(text), shape[0])
        self._env_init = np.zeros(shape, dtype=np.int8)
        for i, line in enumerate(text):
            for j, char in enumerate(line):
                try:
                    self._env_init[i, j] = self._mapper[char]
                except ValueError:
                    # If value not in list
                    # consider as empty space
                    self._env_init[i, j] = TextStates.EMPTY

    @property
    def states_size(self) -> int:
        return np.prod(self._env_init.shape)

    @property
    def actions_size(self) -> int:
        return self._neighbor

    def _render(self) -> None:
        if self._gui is not None:
            data = self._env_exec
            data = np.interp(data, (data.min(), data.max()), (255, 0))
            data = Image.fromarray(data.astype(np.uint8), mode='L').convert('RGBA')
            data = data.resize(size=(data.size[0] * 7, data.size[1] * 7))
            self._gui.append(data)

    def render(self) -> None:
        if self._gui is not None:
            self._gui[0].save(
                datetime.now().strftime("%d-%b-%Y %H:%M:%S.%f") + '.gif',
                save_all=True,
                append_images=self._gui[1:],
                optimize=False,
                duration=75,
                loop=1
            )
        self._gui = []

    def _location_to_state(self, location: List[int]) -> int:
        return location[0] * self._env_exec.shape[0] + location[1]

    def reset(self, *args, **kwargs) -> int:
        # Copy initial env
        self._env_exec = self._env_init.copy()
        # Initialize agent attributes
        agent = {}
        agent['counter'] = 0
        spawn = np.argwhere(self._env_init == TextStates.SPAWN)
        spawn = list(choice(spawn))
        agent['location'] = spawn
        self._agents[kwargs['id']] = agent
        if not self._raw_state:
            return self._location_to_state(spawn)
        return spawn

    def run_step(self, action, *args, **kwargs):
        # Select agent
        agent = self._agents[kwargs['id']]
        # Initialize reward as EMPTY, done as False and wins as 0
        reward = self._rewards[TextStates.EMPTY]
        done = False
        info = {'return_sum': reward, 'wins': 0}
        # Update and check max_steps
        agent['counter'] += 1
        if agent['counter'] == self._max_steps:
            if not self._raw_state:
                return self._location_to_state(agent['location']), reward, True, info
            return agent['location'], reward, True, info
        # Copy the list
        next_state = agent['location'][::]
        if self._neighbor == TextNeighbor.NEUMANN:
            # Compute action mod
            i = (action % 2)
            j = (action < 2) ^ i
            # Update next state (-1, 1, 1, -1)
            next_state[i] += (-1) ** j
        if self._neighbor == TextNeighbor.MOORE:
            raise NotImplementedError
        # Check if next state is valid
        if next_state[i] >= 0 and next_state[i] < self._env_exec.shape[i]:
            # Set pointer to env location
            pointer = self._env_exec[tuple(next_state)]
            # Check if next state is not an AGENT nor a WALL
            if pointer != TextStates.AGENT and pointer != TextStates.WALL:
                # Check if next state is GOAL
                if pointer == TextStates.GOAL:
                    # Set GOAL reward value and done flag
                    reward = self._rewards[TextStates.GOAL]
                    done = True
                    info = {'return_sum': reward, 'wins': 1}
                # Reset agent location
                self._env_exec[tuple(agent['location'])] = TextStates.EMPTY
                # Update agent loaction
                agent['location'] = next_state
                # Set agent on env if not reached GOAL
                if not done:
                    self._env_exec[tuple(next_state)] = TextStates.AGENT
            else:
                # Set reward for WALL
                reward = self._rewards[TextStates.WALL]
        # Render image
        self._render()
        # Return S, R, done, info
        if not self._raw_state:
            return self._location_to_state(agent['location']), reward, done, info
        return agent['location'], reward, done, info

    @property
    def probability_distribution(self):
        return None
