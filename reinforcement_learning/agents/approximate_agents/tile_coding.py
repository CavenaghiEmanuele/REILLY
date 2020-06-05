import numpy as np

from typing import List, Dict


class Tiling:

    _start_point = np.array
    _tiles: Dict
    _tiles_dims: np.array
    _feature_dims: int

    def __init__(self, feature_dims: int, tiles_dims: np.array, start_point: np.array):
        self._start_point = start_point
        self._tiles_dims = tiles_dims
        self._tiles = {}
        self._feature_dims = feature_dims

    def __str__(self):
        return "Tiling start point: " + str(self._start_point) + " - \n" + \
            "Tiles dims: " + str(self._tiles_dims) + " - \n" + \
            "Tiles: " + str(self._tiles) + "\n"

    def __repr__(self):
        return "Tiling start point: " + str(self._start_point) + " - \n" + \
            "Tiles dims: " + str(self._tiles_dims) + " - \n" + \
            "Tiles: " + str(self._tiles) + "\n"

    def add_tiles(self, start_point: np.array, action: int):
        end_point = start_point + self._tiles_dims

        if not str((start_point, end_point, action)) in self._tiles:
            self._tiles.update(
                {str((start_point, end_point, action)): len(self._tiles)})

    def get_tile_index(self, features: List[float], action: int):
        start_point = self.get_lower_bound(features)
        end_point = start_point + self._tiles_dims
        
        if not str((start_point, end_point, action)) in self._tiles:
            self.add_tiles(start_point, action)
        return self._tiles[str((start_point, end_point, action))]

    def get_lower_bound(self, features: List[float]) -> int:
        lower_bound = []
        features= np.absolute(features - self._start_point)
        for i in range(len(features)):
            j = self._start_point[i]
            while features[i] > self._tiles_dims[i]:
                features[i] -= self._tiles_dims[i]
                j += self._tiles_dims[i]

            if features[i] > 0:
                lower_bound.append(j)
            else:
                lower_bound.append(-j)

        return lower_bound


class TileCoding():

    _tilings: List[Tiling]
    _tiling_offset: np.array

    def __init__(self, feature_dims: int, tilings_offset: List[float], tiles_dims: List[float], n_tilings: int = 8):
        
        if tilings_offset == None:
            tilings_offset = np.ones(feature_dims)
        if isinstance(tilings_offset, int):
            tilings_offset *= np.ones(feature_dims)
        if tiles_dims == None:
            tiles_dims = np.ones(feature_dims)
        if isinstance(tiles_dims, int):
            tiles_dims *= np.ones(feature_dims)
        
        self._tiling_offset = np.asarray(tilings_offset)
        self._tilings = [Tiling(
                            feature_dims=feature_dims, 
                            tiles_dims= np.asarray(tiles_dims), 
                            start_point= np.asarray(-i * tilings_offset))
                                for i in range(n_tilings)
                        ]

    def __str__(self):
        s = ""
        for tiling in self._tilings:
            s += "----------------------\n" + str(tiling)
        return "Tilings: \n" + s

    def get_coordinates(self, features: List[float], action: int) -> List[int]:
        return [tiling.get_tile_index(features, action) for tiling in self._tilings]
