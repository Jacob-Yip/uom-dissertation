"""
There will be a Python error thrown if there is a cycle of the graph, which is expected and something we want
"""


class Node:
    def __init__(self, node_id: int, bias=0.0, successors=None):
        """
        NOTE: If argument is set to be successors=[], there will be unexpected behaviour for Node(node_id=...) as multiple instances of Node will share the same list

        :param: successors: The list of Node instances of the successors of this node instance
        """
        self.__node_id = node_id
        self.__bias = bias
        self.__is_start_node = True
        if successors is None:
            self.__successors = []
        else:
            self.__successors = successors

    @property
    def is_start_node(self) -> bool:
        return self.__is_start_node

    @is_start_node.setter
    def is_start_node(self, is_start_node: bool):
        """
        A recursive operation
        """
        self.__is_start_node = is_start_node

        for successor in self.__successors:
            successor.is_start_node = False

    def add_successor(self, successor) -> None:
        self.__successors.append(successor)

        successor.is_start_node = False

    def add_successors(self, successors: list) -> None:
        self.__successors += successors

        for successor in successors:
            successor.is_start_node = False

    @property
    def node_id(self) -> int:
        return self.__node_id

    @property
    def bias(self) -> float:
        return self.__bias

    @property
    def successors(self) -> list:
        return self.__successors

    def __str__(self) -> str:
        return f"{{'node_id': {self.__node_id}, 'bias': {self.__bias}, 'is_start_node': {self.__is_start_node}, 'successors': {[str(successor) for successor in self.__successors]}}}"
