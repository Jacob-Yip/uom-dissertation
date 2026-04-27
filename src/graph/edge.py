
"""
Assume this connection is enabled
If it is disabled, we ignore it
"""


class Edge:
    def __init__(self, key: tuple, weight=0.0):
        self.__key = key
        self.__weight = weight

    @property
    def key(self) -> tuple:
        return self.__key

    @property
    def weight(self) -> float:
        return self.__weight

    def __str__(self) -> str:
        return f"{{'key': {self.__key}, 'weight': {self.__weight}}}"
