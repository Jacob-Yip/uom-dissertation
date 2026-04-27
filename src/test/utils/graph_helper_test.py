import neat
import os
from src.neat.neat_ncl_genome import NEATNCLGenome
import src.utils.graph_helper as graph_helper

"""
Run: python -m src.test.utils.graph_helper_test
"""

# Constant
# XOR input-output pairs
xor_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
xor_outputs = [0.0, 1.0, 1.0, 0.0]
# Load configuration file
# Expect to run at root level
config_path = os.path.join(os.path.dirname(
    __file__), os.path.join("./", "config-test.ini"))
config = neat.Config(
    NEATNCLGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    config_path
)

# ===================================================================================


def eval_genomes(genomes, config):
    """
    Fitness evaluation
    """
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        fitness = 4.0  # Max possible fitness
        for inputs, expected in zip(xor_inputs, xor_outputs):
            output = net.activate(inputs)[0]
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


if __name__ == "__main__":
    print(f"Running test for graph_helper.py ...")

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

    # Create genome grpah

    genome_nodes, genome_edges, genome_activation = graph_helper.get_genome_graph(
        genome=best_genome,
        input_keys=config.genome_config.input_keys
    )

    # ==================================================================================================================

    print(f"======================= Test for get_genome_graph() =========================")
    print(f"Printing genome graph")

    print(f"Nodes: ")
    for node in genome_nodes.values():
        print(f"- {node}")
    print(f"===============================================")
    print(f"Edges: ")
    for edge in genome_edges.values():
        print(f"- {edge}")
    print(f"===============================================")
    print(f"Activation: ")
    print(f"{genome_activation}")

    # ==================================================================================================================

    print(f"======================= Test for topological_sort() =========================")

    sorted_node_ids = graph_helper.topological_sort(nodes=genome_nodes)

    print(f"Topological sort: {sorted_node_ids}")

    print(f"Test for graph_helper.py finishes running ...")
