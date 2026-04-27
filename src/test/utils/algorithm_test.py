from src.graph.edge import Edge
from src.graph.node import Node
from src.model.mlp import MLP
import src.utils.graph_helper as graph_helper
import src.utils.mlp_helper as mlp_helper
import torch
"""
For testing different algorithms
Conclusion
- My algorithm cannot convert a fully-connected network to the network in NEAT format
    - Just test it yourself with mode = 1
    - mode = 0 works because there are never negative values in a node (ReLU is assumed to be the activation function used)

Run: python -m src.test.utils.algorithm_test
"""

# Constant
MODE = 0

# ===================================================================================


def build_graph(mode=0) -> tuple:
    if mode == 0:
        """
        Node_-1: 
        ---- 1 ----> Node_1
        ---- 2 ----> Node_2 (bias = +1)

        Node_-2: 
        ---- 10 ----> Node_2 (bias = +1)
        ---- 20 ----> Node_3

        Node_1: 
        ---- 1 ----> Node_4

        Node_2: 
        ---- 1 ----> Node_4

        Node_3: 
        ---- 1 ----> Node_4

        Topological sort calculated with my own algorithm: -2, -1, 1, 2, 3, 4
        """
        # Input layer
        node_start_0 = Node(
            node_id=-1
        )
        node_start_1 = Node(
            node_id=-2
        )

        # Hidden-1 layer
        node_1 = Node(node_id=1)
        node_2 = Node(node_id=2, bias=1)
        node_3 = Node(node_id=3)

        # Output layer
        node_4 = Node(node_id=4)

        # Layer-1
        node_start_0.add_successor(node_1)
        node_start_0.add_successor(node_2)
        node_start_1.add_successor(node_2)
        node_start_1.add_successor(node_3)

        # Layer-2
        node_1.add_successor(node_4)
        node_2.add_successor(node_4)
        node_3.add_successor(node_4)

        nodes = {
            -2: node_start_1,
            -1: node_start_0,
            1: node_1,
            2: node_2,
            3: node_3,
            4: node_4
        }

        edges = {
            (-1, 1): Edge(key=(-1, 1), weight=1),
            (-1, 2): Edge(key=(-1, 2), weight=2),
            (-2, 2): Edge(key=(-2, 2), weight=10),
            (-2, 3): Edge(key=(-2, 3), weight=20),
            (1, 4): Edge(key=(1, 4), weight=1),
            (2, 4): Edge(key=(2, 4), weight=1),
            (3, 4): Edge(key=(3, 4), weight=1)
        }

        topological_sort = [-2, -1, 1, 2, 3, 4]
    elif mode == 1:
        """
        Node_-1: 
        ---- 1 ----> Node_1

        Node_-2: 
        ---- 2 ----> Node_2 (bias = -1)

        Node_1: 
        ---- 10 ----> Node_3
        ---- 20 ----> Node_4 (bias = +1)

        Node_2: 
        ---- 30 ----> Node_4 (bias = +1)

        Node_3: 
        ---- 1 ----> Node_5

        Node_4: 
        ---- 1 ----> Node_5

        Topological sort calculated with my own algorithm: -2, -1, 1, 2, 3, 4, 5
        """
        # Input layer
        node_start_0 = Node(
            node_id=-1
        )
        node_start_1 = Node(
            node_id=-2
        )

        # Hidden-1 layer
        node_1 = Node(node_id=1)
        node_2 = Node(node_id=2, bias=-1)

        # Hidden-2 layer
        node_3 = Node(node_id=3)
        node_4 = Node(node_id=4, bias=1)

        # Output layer
        node_5 = Node(node_id=5)

        # Layer-1
        node_start_0.add_successor(node_1)
        node_start_1.add_successor(node_2)

        # Layer-2
        node_1.add_successor(node_3)
        node_1.add_successor(node_4)
        node_2.add_successor(node_4)

        # Layer-3
        node_3.add_successor(node_5)
        node_4.add_successor(node_5)

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
            (-1, 1): Edge(key=(-1, 1), weight=1),
            (-2, 2): Edge(key=(-2, 2), weight=2),
            (1, 3): Edge(key=(-2, 3), weight=10),
            (1, 4): Edge(key=(1, 4), weight=20),
            (2, 4): Edge(key=(2, 4), weight=30),
            (3, 5): Edge(key=(3, 5), weight=1),
            (4, 5): Edge(key=(4, 5), weight=1)
        }

        topological_sort = [-2, -1, 1, 2, 3, 4, 5]
    elif mode == 2:
        """
        Node_-1: 
        ---- 10 ----> Node_2

        Node_-2: 
        ---- 30 ----> Node_2

        Node_-3: 
        ---- 60 ----> Node_2
        ---- 70 ----> Node_1

        Node_1: 
        None - output node

        Node_2: 
        ---- 20 ----> Node_1
        ---- 40 ----> Node_3

        Node_3: 
        ---- 50 ----> Node_1

        Topological sort calculated with my own algorithm: -3, -2, -1, 2, 3, 1
        """
        # Input layer
        node_start_0 = Node(
            node_id=-1
        )
        node_start_1 = Node(
            node_id=-2
        )
        node_start_2 = Node(
            node_id=-3
        )

        # Hidden-1 layer
        node_2 = Node(node_id=2)

        # Hidden-2 layer
        node_3 = Node(node_id=3)

        # Output layer
        node_1 = Node(node_id=1)

        # Layer-1
        node_start_0.add_successor(node_2)
        node_start_1.add_successor(node_2)
        node_start_2.add_successor(node_2)
        node_start_2.add_successor(node_1)

        # Layer-2
        node_2.add_successor(node_1)
        node_2.add_successor(node_3)

        # Layer-3
        node_3.add_successor(node_1)

        nodes = {
            -3: node_start_2,
            -2: node_start_1,
            -1: node_start_0,
            1: node_1,
            2: node_2,
            3: node_3
        }

        edges = {
            (-1, 2): Edge(key=(-1, 2), weight=10),
            (-2, 2): Edge(key=(-2, 2), weight=30),
            (-3, 2): Edge(key=(-3, 2), weight=60),
            (-3, 1): Edge(key=(-3, 1), weight=70),
            (2, 1): Edge(key=(2, 1), weight=20),
            (2, 3): Edge(key=(2, 3), weight=40),
            (3, 1): Edge(key=(3, 1), weight=50)
        }

        topological_sort = [-3, -2, -1, 2, 3, 1]
    else:
        raise Exception(f"Invalid mode: {mode}")

    return (nodes, edges, topological_sort)


def get_explicit_linear_layers(linear_layer_mode=0):
    if linear_layer_mode == 0:
        """
        Node_-1: 
        ---- 1 ----> Node_1
        ---- 2 ----> Node_2 (bias = +1)

        Node_-2: 
        ---- 10 ----> Node_2 (bias = +1)
        ---- 20 ----> Node_3

        Node_1: 
        ---- 1 ----> Node_4

        Node_2: 
        ---- 1 ----> Node_4

        Node_3: 
        ---- 1 ----> Node_4

        Topological sort calculated with my own algorithm: -2, -1, 1, 2, 3, 4
        """
        hidden_layer_1 = torch.nn.Linear(2, 3)
        output_layer = torch.nn.Linear(3, 1)

        with torch.no_grad():
            hidden_layer_1.weight.copy_(torch.tensor([
                [1.0, 0.0],  # Node_1 connections
                [2.0, 10.0],  # Node_2 connections
                [0.0, 20.0]  # Node_3 connections
            ]))
            hidden_layer_1.bias.copy_(
                torch.tensor([0.0, 1.0, 0.0]))  # Node_2 bias

            output_layer.weight.copy_(torch.tensor([[1.0, 1.0, 1.0]]))
            output_layer.bias.fill_(0.0)

        return [hidden_layer_1, output_layer]
    elif linear_layer_mode == 1:
        """
        Node_-1: 
        ---- 1 ----> Node_1

        Node_-2: 
        ---- 2 ----> Node_2 (bias = -1)

        Node_1: 
        ---- 10 ----> Node_3
        ---- 20 ----> Node_4 (bias = +1)

        Node_2: 
        ---- 30 ----> Node_4 (bias = +1)

        Node_3: 
        ---- 1 ----> Node_5

        Node_4: 
        ---- 1 ----> Node_5

        Topological sort calculated with my own algorithm: -2, -1, 1, 2, 3, 4, 5
        """
        hidden_layer_1 = torch.nn.Linear(2, 2)
        hidden_layer_2 = torch.nn.Linear(2, 2)
        output_layer = torch.nn.Linear(2, 1)

        with torch.no_grad():
            hidden_layer_1.weight.copy_(torch.tensor([
                [1.0, 0.0],  # Node_1 connections
                [0.0, 2.0]  # Node_2 connections
            ]))
            hidden_layer_1.bias.copy_(
                torch.tensor([0.0, -1.0]))  # Node_2 bias

            hidden_layer_2.weight.copy_(torch.tensor([
                [10.0, 0.0],  # Node_3 connections
                [20.0, 30.0]  # Node_4 connections
            ]))
            hidden_layer_2.bias.copy_(
                torch.tensor([0.0, 1.0]))  # Node_4 bias

            output_layer.weight.copy_(torch.tensor([[1.0, 1.0]]))
            output_layer.bias.fill_(0.0)

        return [hidden_layer_1, hidden_layer_2, output_layer]
    elif linear_layer_mode == 2:
        """
        Node_-1: 
        ---- 10 ----> Node_2

        Node_-2: 
        ---- 30 ----> Node_2

        Node_-3: 
        ---- 60 ----> Node_2
        ---- 70 ----> Node_1

        Node_1: 
        None - output node

        Node_2: 
        ---- 20 ----> Node_1
        ---- 40 ----> Node_3

        Node_3: 
        ---- 50 ----> Node_1

        Topological sort calculated with my own algorithm: -3, -2, -1, 2, 3, 1
        """
        raise Exception(f"Have not implemented yet")
    else:
        raise Exception(f"Invalid linear_layer_mode: {linear_layer_mode}")


def get_data(mode=0) -> tuple:
    """
    Return the data to test the implementation of my algorithm
    data_y is the true value, not the prediction

    :param mode: Test case index
    :return: (data_X, data_y)
    :rtype: tuple
    """
    if mode == 0:
        data_X = torch.tensor([
            [0, 1]  # Expect 31
        ], dtype=torch.float)
        (data_y) = torch.tensor([
            32
        ], dtype=torch.float)
    elif mode == 1:
        # NOTE: It's 31 because we are using the activation function ReLU
        data_X = torch.tensor([
            [1, 0]  # Expect 31
        ], dtype=torch.float)
        (data_y) = torch.tensor([
            32
        ], dtype=torch.float)
    elif mode == 2:
        raise Exception(f"Have not implemented")
    else:
        raise Exception(f"Invalid mode: {mode}")

    return (data_X, data_y)


def test_my_algorithm_backpropagation(mode=0) -> None:
    """
    Test that backpropagation works in the neural networks created by my algorithm
    This makes sure I can apply backpropagation to NEAT neural networks, which do not have a fixed architecture
    For NEAT
    """
    # Constant
    LEARNING_RATE = 0.1

    print(f"Running on mode {mode} ...")

    # Create graph
    nodes, edges, topological_sort = build_graph(mode=mode)

    # Log
    # print(f"Printing genome graph")

    # print(f"Nodes: ")
    # for node in nodes.values():
    #     print(f"- {node}")
    # print(f"===============================================")
    # print(f"Edges: ")
    # for edge in edges.values():
    #     print(f"- {edge}")
    # print(f"===============================================")
    # print(f"Topological sort: ")
    # print(f"{topological_sort}")

    # ==================================================================================================================

    linear_layers = mlp_helper.build_linear_layers(
        topological_sort=topological_sort, nodes=nodes, edges=edges)

    # NOTE: Uncomment the line below to get explicitly defined linear layers
    # linear_layers = get_explicit_linear_layers(linear_layer_mode=mode)

    model = MLP.build_from_linear_layers(
        layers=linear_layers, layer_heights=None, activations=None)

    # Log
    print(f"Number of parameters to optimize: {len(list(model.parameters()))}")

    # SGD is better for inspecting the weights changed due to backpropagation compared to the more complicated optimiser Adam
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    # optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Log
    print(f"Linear layers: ")
    for linear_layer in model.layers:
        print(f"Weight: ")
        print(linear_layer.weight)
        print(f"Bias: ")
        print(linear_layer.bias)
        print(f"---------------------------------------------------------------")

    data_X, data_y = get_data(mode=mode)

    y_prediction = model(data_X)

    print(f"data_X: {data_X}; y_prediction: {y_prediction}; data_y: {data_y}")

    # MSE loss
    mse_loss = torch.mean((data_y - y_prediction) ** 2)

    print(f"MSE Loss: {mse_loss}")

    optimizer.zero_grad()  # Reset gradients
    mse_loss.backward()

    # Apply gradient descent once
    optimizer.step()

    print(f"===========================================================================")

    # Log
    print(f"Updated linear layers: ")
    for linear_layer in model.layers:
        print(f"Weight: ")
        print(linear_layer.weight)
        print(f"Bias: ")
        print(linear_layer.bias)
        print(f"---------------------------------------------------------------")


if __name__ == "__main__":
    test_my_algorithm_backpropagation(mode=MODE)

    print(f"Test for algorithms finishes running ...")
