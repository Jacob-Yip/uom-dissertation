import math
import neat
from neat.math_util import mean
from neat.reproduction import DefaultReproduction
import random

"""
A custom reproduction algorithm for NEAT NCL
"""


class NEATNCLReproduction(DefaultReproduction):
    """
    Implements a customised NEAT-NCL reproduction scheme:
    NCL is run before fitness evaluation
    """

    def __init__(self, config, reporters, stagnation):
        super().__init__(config=config, reporters=reporters, stagnation=stagnation)

    def reproduce(self, config: neat.Config, species: neat.DefaultSpeciesSet, pop_size: int, generation: int) -> dict:
        """
        Override

        :param config: The config instance representing the config file
        :type config: neat.Config
        :param species: The list of species (in total, they form the population to be evolved)
        :type species: neat.DefaultSpeciesSet
        :param pop_size: Population size, i.e. number of base learners
        :type pop_size: int
        :param generation: An index indicating which generation we are at right now
        :type generation: int
        :return: {genome_id: genome_instance}
        :rtype: dict
        """
        # Set innovation tracker for this generation and reset generation-specific tracking
        # This enables same-generation deduplication: if multiple genomes make the same
        # structural mutation this generation, they get the same innovation number
        config.genome_config.innovation_tracker = self.innovation_tracker
        self.innovation_tracker.reset_generation()

        # TODO: I don't like this modification of the species and stagnation objects,
        # because it requires internal knowledge of the objects.

        # Filter out stagnated species, collect the set of non-stagnated
        # species members, and compute their average adjusted fitness.
        # The average adjusted fitness scheme (normalized to the interval
        # [0, 1]) allows the use of negative fitness values without
        # interfering with the shared fitness scheme.
        all_fitnesses = []
        remaining_species = []
        for stag_sid, stag_s, stagnant in self.stagnation.update(species, generation):
            if stagnant:
                self.reporters.species_stagnant(stag_sid, stag_s)
            else:
                all_fitnesses.extend(
                    m.fitness for m in stag_s.members.values())
                remaining_species.append(stag_s)
        # The above comment was not quite what was happening - now getting fitnesses
        # only from members of non-stagnated species.

        # No species left.
        if not remaining_species:
            species.species = {}
            return {}  # was []

        # Find minimum/maximum fitness across the entire population, for use in
        # species adjusted fitness computation.
        min_fitness = min(all_fitnesses)
        max_fitness = max(all_fitnesses)
        # Do not allow the fitness range to be zero, as we divide by it below.
        # TODO: The ``1.0`` below is rather arbitrary, and should be configurable.
        fitness_range = max(1.0, max_fitness - min_fitness)
        for afs in remaining_species:
            # Compute adjusted fitness.
            msf = mean([m.fitness for m in afs.members.values()])
            af = (msf - min_fitness) / fitness_range
            afs.adjusted_fitness = af

        adjusted_fitnesses = [s.adjusted_fitness for s in remaining_species]
        avg_adjusted_fitness = mean(adjusted_fitnesses)  # type: float
        self.reporters.info(
            f"Average adjusted fitness: {avg_adjusted_fitness:.3f}")

        # Compute the number of new members for each species in the new generation.
        previous_sizes = [len(s.members) for s in remaining_species]
        min_species_size = self.reproduction_config.min_species_size
        # Isn't the effective min_species_size going to be max(min_species_size,
        # self.reproduction_config.elitism)? That would probably produce more accurate tracking
        # of population sizes and relative fitnesses... doing. TODO: document.
        min_species_size = max(
            min_species_size, self.reproduction_config.elitism)
        spawn_amounts = self.compute_spawn(adjusted_fitnesses, previous_sizes,
                                           pop_size, min_species_size)
        # Adjust spawn counts so that the total exactly matches the requested
        # population size while respecting the per-species minimum.
        spawn_amounts = self._adjust_spawn_exact(
            spawn_amounts, pop_size, min_species_size)

        new_population = {}
        species.species = {}
        for spawn, s in zip(spawn_amounts, remaining_species):
            # If elitism is enabled, each species always at least gets to retain its elites.
            spawn = max(spawn, self.reproduction_config.elitism)

            assert spawn > 0

            # The species has at least one member for the next generation, so retain it.
            old_members = list(s.members.items())
            s.members = {}
            species.species[s.key] = s

            # Sort members in order of descending fitness, with genome id as a
            # deterministic tie-breaker so that ordering (and thus parent
            # selection) is reproducible across runs and checkpoint restores.
            old_members.sort(reverse=True, key=lambda x: (x[1].fitness, x[0]))

            # Transfer elites to new generation.
            if self.reproduction_config.elitism > 0:
                for i, m in old_members[:self.reproduction_config.elitism]:
                    new_population[i] = m
                    spawn -= 1

            if spawn <= 0:
                continue

            # Only use the survival threshold fraction to use as parents for the next generation.
            repro_cutoff = int(math.ceil(self.reproduction_config.survival_threshold *
                                         len(old_members)))
            # Use at least two parents no matter what the threshold fraction result is.
            repro_cutoff = max(repro_cutoff, 2)
            old_members = old_members[:repro_cutoff]

            # Index of the first parent
            # +1 to get parent_2_index
            # NOTE: old_members will at least have 2 parents
            parent_1_index = 0

            # Choose the best pair of parents and produce the number of offspring allotted to the species.
            while spawn > 0:
                spawn -= 1

                # Get the best pair of paraents
                if parent_1_index >= len(old_members) - 1:
                    # We have reached the worst pair of parents -> go back to the best pair of parents
                    parent_1_index = 0

                parent1_id, parent1 = old_members[parent_1_index]
                # NOTE: The use of % is for the special case where len(old_members) == 1
                parent2_id, parent2 = old_members[(
                    parent_1_index + 1) % len(old_members)]

                parent_1_index += 2

                # Note that if the parents are not distinct, crossover will produce a
                # genetically identical clone of the parent (but with a different ID).
                gid = next(self.genome_indexer)
                child = config.genome_type(gid)
                child.configure_crossover(
                    parent1, parent2, config.genome_config)
                child.mutate(config.genome_config)
                new_population[gid] = child
                self.ancestors[gid] = (parent1_id, parent2_id)

            """
            NOTE: We can apply NCL update here but we do not
            NOTE: This is because the structure of NEAT's code is bounded by neat-Python and I cannot add parameters to the method easily -> it needs a dynamic penalty coefficient for NCL update
            NOTE: So the workaround is to apply NCL update right before the fitness of each genome is evaluated
            """

        return new_population
