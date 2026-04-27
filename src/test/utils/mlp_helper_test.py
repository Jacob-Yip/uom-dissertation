from src.graph.edge import Edge
from src.graph.node import Node
import src.utils.graph_helper as graph_helper
import src.utils.mlp_helper as mlp_helper

"""
Run: python -m src.test.utils.mlp_helper_test
"""

# Constant
mode = 2

# ===================================================================================


def build_graph(mode=0) -> tuple:
    if mode == 0:
        node_start_0 = Node(
            node_id=-1
        )
        node_start_1 = Node(
            node_id=-2
        )

        node_1 = Node(node_id=1)
        node_2 = Node(node_id=2, bias=4)
        node_3 = Node(node_id=3)
        node_4 = Node(node_id=4, bias=8)
        node_5 = Node(node_id=5)

        node_1.add_successor(node_4)
        node_2.add_successor(node_4)
        node_3.add_successor(node_5)

        node_start_0.add_successor(node_1)
        node_start_0.add_successor(node_5)
        node_start_1.add_successor(node_2)
        node_start_1.add_successor(node_3)
        node_start_1.add_successor(node_5)

        nodes = {
            -2: node_start_1,
            -1: node_start_0,
            1: node_1,
            2: node_2,
            3: node_3,
            4: node_4,
            5: node_5
        }

        edges = {
            (-1, 1): Edge((-1, 1), weight=1),
            (1, 4): Edge((1, 4), weight=2),
            (-1, 5): Edge((-1, 5), weight=3),
            (2, 4): Edge((2, 4), weight=4),
            (-2, 2): Edge((-2, 2), weight=5),
            (-2, 3): Edge((-2, 3), weight=6),
            (3, 5): Edge((3, 5), weight=7),
            (-2, 5): Edge((-2, 5), weight=8)
        }

        topological_sort = [-2, 3, 2, -1, 1, 4, 5]
    elif mode == 1:
        node_start_0 = Node(
            node_id=-1
        )
        node_start_1 = Node(
            node_id=-2
        )

        node_1 = Node(node_id=1)
        node_2 = Node(node_id=2, bias=4)

        node_start_0.add_successor(node_1)
        node_start_0.add_successor(node_2)
        node_start_1.add_successor(node_1)

        nodes = {
            -2: node_start_1,
            -1: node_start_0,
            1: node_1,
            2: node_2
        }

        edges = {
            (-1, 1): Edge(key=(-1, 1), weight=1),
            (-1, 2): Edge(key=(-1, 2), weight=2),
            (-2, 1): Edge(key=(-2, 1), weight=3)
        }

        topological_sort = [-2, -1, 2, 1]
    elif mode == 2:
        """
        Node_-1: 
        ---- 1 ----> Node_1
        ---- 2 ----> Node_2 (bias = +1)

        Node_-2: 
        ---- 10 ----> Node_2 (bias = +1)
        ---- 20 ----> Node_3

        Topological sort calculated with my own algorithm: -2, -1, 1, 2, 3
        """
        node_start_0 = Node(
            node_id=-1
        )
        node_start_1 = Node(
            node_id=-2
        )

        node_1 = Node(node_id=1)
        node_2 = Node(node_id=2, bias=1)
        node_3 = Node(node_id=3)

        node_start_0.add_successor(node_1)
        node_start_0.add_successor(node_2)
        node_start_1.add_successor(node_2)
        node_start_1.add_successor(node_3)

        nodes = {
            -2: node_start_1,
            -1: node_start_0,
            1: node_1,
            2: node_2,
            3: node_3
        }

        edges = {
            (-1, 1): Edge(key=(-1, 1), weight=1),
            (-1, 2): Edge(key=(-1, 2), weight=2),
            (-2, 2): Edge(key=(-2, 2), weight=10),
            (-2, 3): Edge(key=(-2, 3), weight=20)
        }

        topological_sort = [-2, -1, 1, 2, 3]
    else:
        raise Exception(f"Invalid mode: {mode}")

    return (nodes, edges, topological_sort)


if __name__ == "__main__":
    print(f"Running test for mlp_helper.py ...")

    # Create graph
    nodes, edges, topological_sort = build_graph(mode=mode)

    print(f"Printing genome graph")

    print(f"Nodes: ")
    for node in nodes.values():
        print(f"- {node}")
    print(f"===============================================")
    print(f"Edges: ")
    for edge in edges.values():
        print(f"- {edge}")
    print(f"===============================================")
    print(f"Topological sort: ")
    print(f"{topological_sort}")

    # ==================================================================================================================

    print(f"======================= Test for build_linear_layers() =========================")

    linear_layers = mlp_helper.build_linear_layers(
        topological_sort=topological_sort, nodes=nodes, edges=edges)

    print(f"Linear layers: ")
    for linear_layer in linear_layers:
        print(f"Weight: ")
        print(linear_layer.weight)
        print(f"Bias: ")
        print(linear_layer.bias)
        print(f"---------------------------------------------------------------")

    print(f"Test for mlp_helper.py finishes running ...")
