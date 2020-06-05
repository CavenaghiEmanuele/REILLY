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

    _env: np.ndarray
    _init: np.ndarray
    _agent: List[int]
    _neighbor: int
    _mapper: Dict = {
        ' ': TextStates.EMPTY,          # Empty space
        'S': TextStates.SPAWN,          # Spawn zones
        'A': TextStates.AGENT,          # Agent pointers
        'X': TextStates.GOAL,           # Goal pointers
        '#': TextStates.WALL,           # Walls
    }

    def __init__(self, text: str, neighborhood: int = TextNeighbor.MOORE):
        # Set neighborhood type
        self._neighbor = neighborhood
        # Remove formatting chars at beginning or end of text
        text = text.strip('\n\r\t').split('\n')
        # Check if all lines have equal length
        shape = [len(line) for line in text]
        if len(set(shape)) > 1:
            raise ValueError
        # Compute env shape
        shape = (len(text), shape[0])
        self._init = np.zeros(shape, dtype=np.int8)
        for i, line in enumerate(text):
            for j, char in enumerate(line):
                try:
                    self._init[i, j] = self._mapper[char]
                except ValueError:
                    # If value not in list
                    # consider as empty space
                    self._init[i, j] = TextStates.EMPTY

    @property
    def states_size(self) -> int:
        return np.prod(self._init.shape)

    @property
    def actions_size(self) -> int:
        return self._neighbor

    def render(self) -> None:
        return None

    def reset(self) -> int:
        self._env = self._init.copy()
        spawn = np.argwhere(self._env == TextStates.SPAWN)
        spawn = list(choice(spawn))
        self._agent = spawn
        spawn = spawn[0] * self._env.shape[0] + spawn[1]
        return spawn

    def run_step(self, action, *args, **kwargs):
        # Initialize reward as -1 done as False
        reward = -1
        done = False
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
                        reward = 10
                        done = True
                    # Update agent loaction
                    self._agent = next_state
                    # Compute linear value of next state
                    next_state = next_state[0] * self._env.shape[0] + next_state[1]
                    # Return S, R, done, info
                    return next_state, reward, done, None
        if self._neighbor == TextNeighbor.MOORE:
            raise NotImplementedError
        return (self._agent[0] * self._env.shape[0] + self._agent[1]), reward, done, None

    @property
    def probability_distribution(self):
        return None
