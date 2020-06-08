import numpy as np

from enum import IntEnum, auto, unique
from random import choice
from typing import Dict, List

from ..environment import Environment
from ...utils import Screen


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
        '_gui', '_env', '_agent', '_counter',
        '_neighbor', '_max_steps', '_mapper'
    ]

    _gui: Screen
    _env: np.ndarray
    _agent: List[int]
    _counter: int
    _neighbor: int
    _max_steps: int
    _mapper: Dict

    def __init__(self,
        text: str = '##########\n          \n S        \n          \n        X \n          \n##########',
        neighbor: int = TextNeighbor.MOORE,
        max_steps: int = 50
    ):
        # Set default GUI to None
        self._gui = None
        # Set neighborhood type
        self._neighbor = neighbor
        # Set maximun steps for agent exploration
        self._max_steps = max_steps
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
        self._env = np.zeros(shape, dtype=np.int8)
        for i, line in enumerate(text):
            for j, char in enumerate(line):
                try:
                    self._env[i, j] = self._mapper[char]
                except ValueError:
                    # If value not in list
                    # consider as empty space
                    self._env[i, j] = TextStates.EMPTY

    @property
    def states_size(self) -> int:
        return np.prod(self._env.shape)

    @property
    def actions_size(self) -> int:
        return self._neighbor

    def _render(self) -> None:
        if self._gui:
            data = self._env.copy()
            data[tuple(self._agent)] = TextStates.AGENT
            data = np.interp(data, (data.min(), data.max()), (255, 0))
            self._gui.update(data)

    def render(self) -> None:
        self._gui = Screen(title=self.__class__.__name__)
    
    def _location_to_state(self, location: List[int]) -> int:
        return location[0] * self._env.shape[0] + location[1]

    def reset(self) -> int:
        self._counter = 0
        spawn = np.argwhere(self._env == TextStates.SPAWN)
        spawn = list(choice(spawn))
        self._agent = spawn
        return self._location_to_state(spawn)

    def run_step(self, action, *args, **kwargs):
        # Initialize reward as -1, done as False and wins as 0
        reward = -1
        done = False
        info = {'return_sum': reward, 'wins': 0}
        # Update and check max_steps
        self._counter += 1
        if self._counter == self._max_steps:
            return self._location_to_state(self._agent), reward, True, info
        # Copy the list
        next_state = self._agent[::]
        if self._neighbor == TextNeighbor.NEUMANN:
            # Compute action mod
            i = (action % 2)
            j = (action < 2) ^ i
            # Update next state (-1, 1, 1, -1)
            next_state[i] += (-1) ** j
        if self._neighbor == TextNeighbor.MOORE:
            raise NotImplementedError
        # Set pointer to env location
        pointer = tuple(next_state)
        # Check if next state is valid
        if next_state[i] >= 0 and next_state[i] < self._env.shape[i]:
            # Check if next state is not a WALL
            if self._env[pointer] != TextStates.WALL:
                # Check if next state is GOAL
                if self._env[pointer] == TextStates.GOAL:
                    # Set GOAL reward value and done flag
                    reward = 20
                    done = True
                    info = {'return_sum': reward, 'wins': 1}
                # Update agent loaction
                self._agent = next_state
        # Render image
        self._render()
        # Return S, R, done, info
        return self._location_to_state(self._agent), reward, done, info

    @property
    def probability_distribution(self):
        return None
