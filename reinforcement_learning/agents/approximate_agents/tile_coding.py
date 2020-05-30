from typing import List, Dict
from itertools import zip_longest
from math import floor


class Tiling:

    _start_point = List[float]
    _tiles: Dict
    _tiles_dims: List[float]

    def __init__(self, tiles_dims: List[float], start_point: List[int]):
        self._start_point = start_point
        self._tiles_dims = tiles_dims
        self._tiles = {}

    def __str__(self):
        return "Tiling start point: " + str(self._start_point) + " - \n" + \
            "Tiles dims: " + str(self._tiles_dims) + " - \n" + \
            "Tiles: " + str(self._tiles) + "\n"

    def __repr__(self):
        return "Tiling start point: " + str(self._start_point) + " - \n" + \
            "Tiles dims: " + str(self._tiles_dims) + " - \n" + \
            "Tiles: " + str(self._tiles) + "\n"

    def add_tiles(self, start_point: List[float]):
        end_point = [start_point[i] + self._tiles_dims[i]
                     for i in range(len(self._tiles_dims))]

        if not str((start_point, end_point)) in self._tiles:
            self._tiles.update(
                {str((start_point, end_point)): len(self._tiles)})

    def get_tile_index(self, features: List[float]):

        start_point = self.get_lower_bound(features)
        end_point = [start_point[i] + self._tiles_dims[i]
                     for i in range(len(features))]
        if not str((start_point, end_point)) in self._tiles:
            self.add_tiles(start_point)
        return self._tiles[str((start_point, end_point))]

    def get_lower_bound(self, features: List[float]) -> int:
        lower_bound = []
        for i in range(len(features)):
            feature = abs(features[i] - self._start_point[i])
            j = self._start_point[i]
            while feature > self._tiles_dims[i]:
                feature -= self._tiles_dims[i]
                j += self._tiles_dims[i]

            if features[i] > 0:
                lower_bound.append(j)
            else:
                lower_bound.append(-j)

        return lower_bound


class TileCoding():

    _tilings: List[Tiling]
    _tiling_offset: List[float]

    def __init__(self, n_tilings: int, tilings_offset: List[float], tiles_dims: List[float]):
        self._tiling_offset = tilings_offset
        self._tilings = [Tiling(tiles_dims=tiles_dims, start_point=[-i * offset for offset in tilings_offset])
                         for i in range(n_tilings)]

    def __str__(self):
        s = ""
        for tiling in self._tilings:
            s += "----------------------\n" + str(tiling)
        return "Tilings: \n" + s

    def get_coordinates(self, features: List[float]) -> List[int]:
        return [tiling.get_tile_index(features) for tiling in self._tilings]
