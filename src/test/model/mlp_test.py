import neat
import os
from src.graph.edge import Edge
from src.graph.node import Node
from src.model.mlp import MLP
from src.neat.neat_ncl_genome import NEATNCLGenome
import src.utils.mlp_helper as mlp_helper
import torch


"""
Run: python -m src.test.model.mlp_test
"""

# Constant
# XOR input-output pairs
xor_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
xor_outputs = [0.0, 1.0, 1.0, 0.0]
# Load configuration file
config_path = os.path.join(os.path.dirname(
    __file__), os.path.join("./", "config-test.ini"))
config = neat.Config(
    NEATNCLGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    config_path
)
mode = 0

# ===================================================================================


def eval_genomes(genomes, config):
    """
    Fitness evaluation
    """
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        fitness = 4.0  # Max possible fitness
        for xor_input, expected in zip(xor_inputs, xor_outputs):
            output = net.activate(xor_input)[0]
            fitness -= abs(output - expected)  # Penalize errors
        genome.fitness = fitness


def log_genome(genome: NEATNCLGenome) -> None:
    print(f"=================== Genome ===================")

    print(f"Node: ")
    for (node_id, node) in genome.nodes.items():
        print(
            f"ID: {node_id}; Bias: {node.bias}; Response: {node.response}; Activation: {node.activation}")

    print(f"=====================================")

    print(f"Connection: ")
    for (connection_id, connection) in genome.connections.items():
        print(
            f"ID: {connection_id}; Key: {connection.key}; Weight: {connection.weight}; Enabled: {connection.enabled}")

    print(f"=====================================")


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
    else:
        raise Exception(f"Invalid mode: {mode}")

    return (nodes, edges, topological_sort)


if __name__ == "__main__":
    print(f"Running test for mlp.py ...")

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

    print(f"======================= Test for MLP.build_from_linear_layers() =========================")

    linear_layers = mlp_helper.build_linear_layers(
        topological_sort=topological_sort, nodes=nodes, edges=edges)

    print(f"Linear layers: ")
    for linear_layer in linear_layers:
        print(linear_layer)
        print(f"---------------------------------------------------------------")

    model = MLP.build_from_linear_layers(
        layers=linear_layers, layer_heights=None, activations=None)

    print(f"Linear Layers of this model: ")
    layer_heights = model.layer_heights
    for layer in model.layers:
        print(f"Actual layer: {layer}")
        print(f"---------------------------------------------------------------")
    print(f"Expected layer height: {layer_heights}")

    # ==================================================================================================================

    print(f"======================= Test for MLP.build_from_genome() =========================")

    # Create the population
    population = neat.Population(config)

    # Add console output for progress
    # population.add_reporter(neat.StdOutReporter(True))
    # Can be used later, e.g. data visualization using matplotlib
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    # Run NEAT for up to n generations
    epoch = 50
    best_genome = population.run(eval_genomes, epoch)

    print(f"Printing internal structure of best_genome")
    log_genome(genome=best_genome)

    actual_model = MLP.build_from_genome(genome=best_genome, config=config)
    expected_model = neat.nn.FeedForwardNetwork.create(best_genome, config)

    print(f"Linear Layers of best_genome: ")
    layer_heights = actual_model.layer_heights
    for layer in actual_model.layers:
        print(f"Actual layer: {layer}")
        print(f"Weight of this layer: ")
        print(layer.weight.data)
        print(f"Bias of this layer")
        print(layer.bias.data)
        print(f"---------------------------------------------------------------")
    print(f"Expected layer height: {layer_heights}")

    # Test prediction
    for xor_input, _ in zip(xor_inputs, xor_outputs):
        print(f"NEAT input: {xor_input}")
        torch_input = torch.tensor(xor_input)
        print(f"Torch input: {torch_input}")

        # We are rounding the numbers to 6 significant figures they might be slightly different due to different float precision in neat-python and PyTorch
        expected_output = float("{:.6g}".format(
            expected_model.activate(xor_input)[0]))
        actual_output = float("{:.6g}".format(
            actual_model(torch_input).item()))

        print(f"actual_output: {actual_output}")
        print(f"expected_output: {expected_output}")

        """
        OLD_TODO: Understand why neat model acts weird
        Activation functions must be indempotent for this algorithm to work
        """
        assert expected_output == actual_output, f"Mismatch predictions: {expected_output} != {actual_output}"

        print(f"---------------------------------------------------------------")

    print(f"Congratualations! actual_model == expected_model! ")

    print(f"Test for mlp_helper.py finishes running ...")
