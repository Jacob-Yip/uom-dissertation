import neat
from neat.genome import DefaultGenome
from src.neat.neat_ncl_genes import NEATNCLConnectionGene, NEATNCLNodeGene
from src.neat.neat_ncl_genome_config import NEATNCLGenomeConfig

"""
For now, it is identical to code from neat-python
"""


class NEATNCLGenome(DefaultGenome):
    @classmethod
    def parse_config(cls, param_dict):
        """
        Override
        """
        param_dict['node_gene_type'] = NEATNCLNodeGene
        param_dict['connection_gene_type'] = NEATNCLConnectionGene
        return NEATNCLGenomeConfig(param_dict, cls.__name__)

    def __init__(self, key):
        super().__init__(key=key)

    def distance(self, other, config):
        """
        Override

        # TODO: To be confirmed
        Formula: 
            genetic_distance = (homologous_node_distance_sum + compatibility_disjoint_coefficient * disjoint_node_num) / (max_genome_node_num) + (homologous_connection_distance_sum + compatibility_disjoint_coefficient * disjoint_connection_num) / (max_genome_connection_num)
        """
        # TODO: A temporary fix of a bug
        compatibility_disjoint_coefficient = 0
        if isinstance(config, NEATNCLGenomeConfig):
            compatibility_disjoint_coefficient = config.compatibility_disjoint_coefficient
        elif isinstance(config, neat.Config):
            compatibility_disjoint_coefficient = config.genome_config.compatibility_disjoint_coefficient
        else:
            raise Exception(
                f"Invalid type of config (expect NEATNCLGenomeConfig/neat.Config) used to access compatibility_disjoint_coefficient: {type(config)}")

        # Compute node gene distance component.
        node_distance = 0.0
        if self.nodes or other.nodes:
            disjoint_nodes = 0
            for k2 in other.nodes:
                if k2 not in self.nodes:
                    disjoint_nodes += 1

            for k1, n1 in self.nodes.items():
                n2 = other.nodes.get(k1)
                if n2 is None:
                    disjoint_nodes += 1
                else:
                    # Homologous genes compute their own distance value.
                    node_distance += n1.distance(n2, config)

            max_nodes = max(len(self.nodes), len(other.nodes))
            node_distance = (node_distance +
                             (compatibility_disjoint_coefficient *
                              disjoint_nodes)) / max_nodes

        # Compute connection gene differences.
        connection_distance = 0.0
        if self.connections or other.connections:
            disjoint_connections = 0
            for k2 in other.connections:
                if k2 not in self.connections:
                    disjoint_connections += 1

            for k1, c1 in self.connections.items():
                c2 = other.connections.get(k1)
                if c2 is None:
                    disjoint_connections += 1
                else:
                    # Homologous genes compute their own distance value.
                    connection_distance += c1.distance(c2, config)

            max_conn = max(len(self.connections), len(other.connections))
            connection_distance = (connection_distance +
                                   (compatibility_disjoint_coefficient *
                                    disjoint_connections)) / max_conn

        distance = node_distance + connection_distance
        return distance
