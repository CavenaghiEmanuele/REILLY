import numpy as np

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
        '_env', '_agent', '_render', '_counter',
        '_neighbor', '_max_steps', '_mapper'
    ]

    _env: np.ndarray
    _agent: List[int]
    _render: bool
    _counter: int
    _neighbor: int
    _max_steps: int
    _mapper: Dict

    def __init__(self, text: str, neighbor: int = TextNeighbor.MOORE, max_steps: int = 50):
        # Set default render to False
        self._render = False
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

    def _show(self) -> None:
        # TODO: Update Qt gui
        data = self._env.copy()
        data[tuple(self._agent)] = TextStates.AGENT
        print(data)

    def render(self) -> None:
        # TODO: Init Qt gui
        self._render = True

    def reset(self) -> int:
        self._counter = 0
        spawn = np.argwhere(self._env == TextStates.SPAWN)
        spawn = list(choice(spawn))
        self._agent = spawn
        spawn = spawn[0] * self._env.shape[0] + spawn[1]
        return spawn

    def _location_to_state(self, location: List[int]) -> int:
        return location[0] * self._env.shape[0] + location[1]

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
                    # Check if render
                    if self._render:
                        self._show()
                    # Return S, R, done, info
                    return self._location_to_state(next_state), reward, done, info
        if self._neighbor == TextNeighbor.MOORE:
            raise NotImplementedError
        return self._location_to_state(self._agent), reward, done, info

    @property
    def probability_distribution(self):
        return None
