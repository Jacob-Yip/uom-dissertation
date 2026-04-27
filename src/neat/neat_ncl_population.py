from neat.population import CompleteExtinctionException, Population
from src.neat.neat_ncl_trainer import check_fitness_function_arguments

"""
A customised population
We need this only because I want to be able to pass arguments to the fitness function
"""


class NEATNCLPopulation(Population):
    def __init__(self, config, initial_state=None, seed=None):
        super().__init__(config=config, initial_state=initial_state, seed=seed)

        self.__max_population_diversity = float(0)
        self.__max_population_diversity_experiment = float(0)
        # (Assume is_experiment = True) In the form of {generation_index: experiment_data_dict}
        self.__experiment_data = {}

    def run(self, fitness_function, n=None, fitness_function_arguments={}):
        """
        Override
        """
        # Check if fitness_function_arguments have all the required arguments
        check_fitness_function_arguments(
            fitness_function_arguments=fitness_function_arguments)

        if self.config.no_fitness_termination and (n is None):
            raise RuntimeError(
                "Cannot have no generational limit with no fitness termination")

        k = 0
        while n is None or k < n:
            k += 1

            fitness_function_arguments["max_population_diversity"] = self.__max_population_diversity
            if fitness_function_arguments["is_experiment"]:
                fitness_function_arguments["max_population_diversity_experiment"] = self.__max_population_diversity_experiment

            self.reporters.start_generation(self.generation)

            # Evaluate all genomes using the user-provided function.
            fitness_function_outputs = fitness_function(list(self.population.items()), self.config,
                                                        fitness_function_arguments=fitness_function_arguments)
            self.__max_population_diversity = fitness_function_outputs["max_population_diversity"]
            if fitness_function_arguments["is_experiment"]:
                # NOTE: self.generation starts from 0
                self.__experiment_data[self.generation] = fitness_function_outputs["experiment_data"]
                self.__max_population_diversity_experiment = fitness_function_outputs[
                    "max_population_diversity_experiment"]

            # Gather and report statistics.
            best = None
            for g in self.population.values():
                if g.fitness is None:
                    raise RuntimeError(
                        f"Fitness not assigned to genome {g.key}")

                if best is None or g.fitness > best.fitness:
                    best = g
            self.reporters.post_evaluate(
                self.config, self.population, self.species, best)

            # Track the best genome ever seen.
            if self.best_genome is None or best.fitness > self.best_genome.fitness:
                self.best_genome = best

            # NOTE: My version
            # fv = self.fitness_criterion(
            #     g.fitness for g in self.population.values())

            if not self.config.no_fitness_termination:
                # Original
                # End if the fitness threshold is reached.
                fv = self.fitness_criterion(
                    g.fitness for g in self.population.values())
                if fv >= self.config.fitness_threshold:
                    self.reporters.found_solution(
                        self.config, self.generation, best)
                    break

            # Create the next generation from the current generation.
            self.population = self.reproduction.reproduce(self.config, self.species,
                                                          self.config.pop_size, self.generation)

            # Check for complete extinction.
            if not self.species.species:
                self.reporters.complete_extinction()

                # If requested by the user, create a completely new population,
                # otherwise raise an exception.
                if self.config.reset_on_extinction:
                    self.population = self.reproduction.create_new(self.config.genome_type,
                                                                   self.config.genome_config,
                                                                   self.config.pop_size)
                else:
                    raise CompleteExtinctionException()

            # Divide the new population into species.
            self.species.speciate(
                self.config, self.population, self.generation)

            self.reporters.end_generation(
                self.config, self.population, self.species)

            self.generation += 1

        if self.config.no_fitness_termination:
            self.reporters.found_solution(
                self.config, self.generation, self.best_genome)

        # Added by me
        # I need to make sure all new children genomes have fitness set so that they can be used
        for final_genome in self.population.values():
            if final_genome.fitness is None:
                # Default value set by me - does not matter cause we will no longer train it
                final_genome.fitness = float("-inf")

        return self.best_genome

    @property
    def max_population_diversity(self) -> float:
        return self.__max_population_diversity

    @property
    def experiment_data(self) -> dict:
        """
        If is_experiment = True, this will be a dict in the form of {generation_index: experiment_data}
        NOTE: generation_index starts from 0 instead of 1

        :return: {generation_index: {data_name: data_value}}
        :rtype: dict
        """
        if self.__experiment_data == {}:
            # No experiment data are collected for all generations
            return None
        else:
            return self.__experiment_data
