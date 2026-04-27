from neat.genome import DefaultGenomeConfig

"""
Created because they have a bug in their code

Causes of bug: 
- Change ```config.compatibility_disjoint_coefficient``` to ```config.genome_config.compatibility_disjoint_coefficient```
    - In ```genomes.py```
- Change ```config.compatibility_weight_coefficient``` to ```config.genome_config.compatibility_weight_coefficient```
    - In ```genes.py```
"""


class NEATNCLGenomeConfig(DefaultGenomeConfig):
    def __init__(self, params, section_name='NEATNCLGenome'):
        super().__init__(params=params, section_name=section_name)
