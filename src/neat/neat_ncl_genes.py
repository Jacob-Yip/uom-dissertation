import neat
from neat.genes import DefaultConnectionGene, DefaultNodeGene
from src.neat.neat_ncl_genome_config import NEATNCLGenomeConfig

"""
For now, it is identical to code from neat-python
"""


class NEATNCLNodeGene(DefaultNodeGene):
    def __init__(self, key):
        super().__init__(key=key)

    def distance(self, other, config):
        """
        Override
        """
        # TODO: A temporary fix of a bug
        compatibility_weight_coefficient = 0
        if isinstance(config, NEATNCLGenomeConfig):
            compatibility_weight_coefficient = config.compatibility_weight_coefficient
        elif isinstance(config, neat.Config):
            compatibility_weight_coefficient = config.genome_config.compatibility_weight_coefficient
        else:
            raise Exception(
                f"Invalid type of config (expect NEATNCLGenomeConfig/neat.Config) used to access compatibility_weight_coefficient: {type(config)}")

        d = abs(self.bias - other.bias) + abs(self.response - other.response)
        if self.activation != other.activation:
            d += 1.0
        if self.aggregation != other.aggregation:
            d += 1.0
        return d * compatibility_weight_coefficient


class NEATNCLConnectionGene(DefaultConnectionGene):
    def __init__(self, key, innovation=None):
        super().__init__(key=key, innovation=innovation)

    def distance(self, other, config):
        """
        Override
        """
        # TODO: A temporary fix of a bug
        compatibility_weight_coefficient = 0
        if isinstance(config, NEATNCLGenomeConfig):
            compatibility_weight_coefficient = config.compatibility_weight_coefficient
        elif isinstance(config, neat.Config):
            compatibility_weight_coefficient = config.genome_config.compatibility_weight_coefficient
        else:
            raise Exception(
                f"Invalid type of config (expect NEATNCLGenomeConfig/neat.Config) used to access compatibility_weight_coefficient: {type(config)}")

        d = abs(self.weight - other.weight)
        if self.enabled != other.enabled:
            d += 1.0
        return d * compatibility_weight_coefficient
