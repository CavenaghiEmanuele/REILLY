import numpy as np

from enum import IntEnum, auto, unique
from typing import Dict, Tuple

from ..environment import Environment


class TextStatesType(IntEnum):
    EMPTY = 0
    SPAWN = auto()
    AGENT = auto()
    GOAL = auto()
    WALL = auto()


class TextNeighborhoodType(IntEnum):
    NEUMANN = 4     # North, East, South, West
    MOORE = 8       # N, NE, E, SE, S, SW, W, NW


class TextEnvironment(Environment):

    _env: np.ndarray
    _init: np.ndarray
    _agent: Tuple
    _neighbor: int
    _mapper: Dict = {
        ' ': TextStatesType.EMPTY,          # Empty space
        'S': TextStatesType.SPAWN,          # Spawn zones
        'A': TextStatesType.AGENT,          # Agent pointers
        'X': TextStatesType.GOAL,           # Goal pointers
        '#': TextStatesType.WALL,           # Walls
    }

    def __init__(self, text: str, neighborhood: int = TextNeighborhoodType.MOORE):
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
                    self._init[i, j] = TextStatesType.EMPTY

    @property
    def states_size(self) -> int:
        return np.prod(self._init.shape)

    @property
    def actions_size(self) -> int:
        return self._neighbor

    @abstractmethod
    def render(self) -> None:
        pass

    @abstractmethod
    def reset(self) -> int:
        pass

    @abstractmethod
    def run_step(self, action, *args, **kwargs):
        pass

    @property
    @abstractmethod
    def probability_distribution(self):
        pass
