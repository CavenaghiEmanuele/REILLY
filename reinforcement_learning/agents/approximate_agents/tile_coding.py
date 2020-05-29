from typing import List, Dict
from itertools import zip_longest
from math import floor


class Tiling:

    # _start_point = [0, 0, ..., 0]
    _tiles: Dict
    _tiles_dims: List[float]

    def __init__(self, tiles_dims: List[float]):
        self._tiles_dims = tiles_dims
        self._tiles = {}

    def __str__(self):
        return "Tiles dims: " + str(self._tiles_dims) + " - " + \
            "Tiles: " + str(self._tiles)

    def __repr__(self):
        return "Tiles dims: " + str(self._tiles_dims) + " - " + \
            "Tiles: " + str(self._tiles) + "\n"

    def add_tiles(self, start_point: List[float]):
        end_point = [start_point[i] + self._tiles_dims[i]
                     for i in range(len(self._tiles_dims))]

        if not str((start_point, end_point)) in self._tiles:
            self._tiles.update(
                {str((start_point, end_point)): len(self._tiles)})

    def get_tile_index(self, feature: List[float]):
        '''
        Prendo le feature in ingresso e ritorno l'indice delle tile che 
        contiene quelle feature
        
        Se non c'è nel dizionario quella tile la creo al momento e la aggiungo 
        al dizionario. Il suo start_point sarà l'intero più vicino secondo la
        dimensione delle tile partendo a contare dalla coordinata [0, 0, ..., 0]
        '''
        pass


class TileCoding():

    _tilings: List[Tiling]
    _tiling_offset: List[float]

    def __init__(self, n_tilings: int, tilings_offset: List[float], tiles_dims: List[float]):
        self._tiling_offset = tilings_offset
        self._tilings = [Tiling(tiles_dims=tiles_dims)
                         for i in range(n_tilings)]

    def __str__(self):
        return "Tilings:\n" + str(self._tilings)
    
    
    def get_coordinates(self, features: List[float]):
        
        '''
        Per ogni tiling passo le feature e ogni tiling mi restituirà l'indice 
        della tile che contiene quelle feature (solo una le può contenere)
        
        Restituirò una lista con gli indici di ogni tile, 
        una lista di n_tiling indici (uno per tiling)
        
        Devo semplicemente passare ad ogni tiling le feature e aspettare risposta
        '''
        pass
