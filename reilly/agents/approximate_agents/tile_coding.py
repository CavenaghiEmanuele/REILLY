from math import floor, copysign
from typing import List, Dict


class Tiling:

    _start_point = List
    _tiles: Dict
    _tiles_size: List
    _feature_dims: int

    def __init__(self, feature_dims: int, tiles_size: List, start_point: List):
        self._start_point = start_point
        self._tiles_size = tiles_size
        self._tiles = {}
        self._feature_dims = feature_dims

    def __str__(self):
        return "Tiling start point: " + str(self._start_point) + " - \n" + \
            "Tiles dims: " + str(self._tiles_size) + " - \n" + \
            "Tiles: " + str(self._tiles) + "\n"

    def __repr__(self):
        return "Tiling start point: " + str(self._start_point) + " - \n" + \
            "Tiles dims: " + str(self._tiles_size) + " - \n" + \
            "Tiles: " + str(self._tiles) + "\n"

    def add_tiles(self, start_point: List, action: int):
        end_point = [start_point[i] + self._tiles_size[i]
                     for i in range(self._feature_dims)]
        if str((start_point, end_point, action)) not in self._tiles:
            self._tiles.update(
                {str((start_point, end_point, action)): len(self._tiles)})

    def get_tile_index(self, features: List, action: int):
        start_point = self.get_lower_bound(features)
        end_point = [start_point[i] + self._tiles_size[i]
                     for i in range(self._feature_dims)]
        try:
            return self._tiles[str((start_point, end_point, action))]
        except BaseException:
            self.add_tiles(start_point, action)
            return len(self._tiles)

    def get_lower_bound(self, features: List) -> int:
        return [
            copysign(
                floor(abs(features[i] - self._start_point[i]
                          ) / self._tiles_size[i]),
                features[i])
            for i in range(self._feature_dims)]


class TileCoding():

    _tilings: List[Tiling]
    _tiling_offset: List

    def __init__(self, feature_dims: int, tilings_offset: List[float], tiles_size: List[float], n_tilings: int = 8):

        if tilings_offset is None:
            tilings_offset = [1 for _ in range(feature_dims)]
        if isinstance(tilings_offset, int):
            tilings_offset = [tilings_offset for _ in range(feature_dims)]
        if tiles_size is None:
            tiles_size = [1 for _ in range(feature_dims)]
        if isinstance(tiles_size, int):
            tiles_size = [tiles_size for _ in range(feature_dims)]

        if len(tiles_size) != feature_dims:
            raise Exception("Size of tiles must have the same dimensions of feature dimensions")
        if len(tilings_offset) != feature_dims:
            raise Exception("Tilings offset must have the same dimensions of feature dimensions")

        self._tilings = [Tiling(
            feature_dims=feature_dims,
            tiles_size=tiles_size,
            start_point=[-i * tilings_offset[j] for j in range(feature_dims)])
            for i in range(n_tilings)
        ]

    def __str__(self):
        s = ""
        for tiling in self._tilings:
            s += "----------------------\n" + str(tiling)
        return "Tilings: \n" + s

    def get_coordinates(self, features: List[float], action: int) -> List[int]:
        return [tiling.get_tile_index(features, action) for tiling in self._tilings]
