from typing import Dict


class IndexHashTable:

    _size: int
    _overfull: False
    _dictionary: Dict

    def __init__(self, size):
        self._size = size
        self._overfull = False
        self._dictionary = {}

    def __str__(self):
        "Prepares a string for printing whenever this object is printed"
        return "Collision table:" + \
               " size:" + str(self._size) + " - " + \
               "overfullCount:" + str(self._overfull) + " - " + \
               "dictionary:" + str(len(self._dictionary)) + " items" + \
               "\n" + str(self._dictionary)

    def count_elements(self) -> int:
        return len(self._dictionary)

    def is_full(self) -> bool:
        return len(self._dictionary) >= self._size

    def get_index(self, obj: tuple):
        if obj in self._dictionary:
            return self._dictionary[obj]
        return None

    def add_element(self, obj: tuple) -> None:
        if not self.is_full() and obj not in self._dictionary:
            self._dictionary[obj] = self.count_elements()
            return
        self._overfull = True
