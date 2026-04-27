import neat
from src.neat.neat_ncl_genome import NEATNCLGenome


def log_genome(genome: NEATNCLGenome) -> None:
    print(f"=================== Genome ===================")

    print(f"Node: ")
    for (node_id, node) in genome.nodes.items():
        print(
            f"ID: {node_id}; Bias: {node.bias}; Response: {node.response}; Activation: {node.activation}")

    print(f"-------------------------------------")

    print(f"Connection: ")
    for (connection_id, connection) in genome.connections.items():
        print(
            f"ID: {connection_id}; Key: {connection.key}; Weight: {connection.weight}; Enabled: {connection.enabled}")

    print(f"=====================================")
