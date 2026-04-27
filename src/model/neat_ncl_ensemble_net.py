import neat
import os
import torch
from src.model.neat_ncl_base_learner import NEATNCLBaseLearner
from src.neat.neat_ncl_population import NEATNCLPopulation
from src.neat.neat_ncl_reproduction import NEATNCLReproduction
from src.neat.neat_ncl_trainer import evaluate_genome
from src.voter.voter import Voter
from src.voter.voter_arithmetic_mean import ArithmeticMeanVoter
from src.voter.voter_median import MedianVoter

"""
An ensemble model that uses NEAT and NCL
Expect to run on 1 CPU (but multiple cores) only (NEAT cannot be run on GPU)
    - This is because my ensemble learning requires communication between base learners, which introduces network delays so it is not a good idea to use multiple nodes for training
    - 1 node with multiple cores should be the best
"""


class NEATNCLEnsembleNet:
    def __init__(self, config: neat.Config, model_repository_path="data/model/neat", voter="arithmetic_mean"):
        """
        NOTE: Expect base_learner_num = len(genomes) = config.pop_size

        :param config: The configuration instance
        :type config: neat.Config
        :param model_repository_path: Absolute path to model repository
        :type model_repository_path: str
        :param voter: The type of voter used
            - arithmetic_mean (default)
            - median
            - nn
        """
        assert config is not None, f"Missing required argument config: {config}"

        self.__cpu_core_num = os.cpu_count()

        assert self.__cpu_core_num > 0, f"Invalid number of CPU cores (expect > 0): {self.__cpu_core_num}"

        self.__config = config
        self.reset()

        self.__max_fitness = self.__config.fitness_threshold
        # TODO: Make sure the value range is the same as the one set by neat-python
        assert self.__max_fitness > 0, f"Invalid value of fitness_threshold (expect > 0): {self.__max_fitness}"
        self.__base_learner_num = self.__config.pop_size
        assert self.__base_learner_num > 0, f"Invalid number of base learners (expect > 1): {self.__base_learner_num}"
        self.__model_repository_path = model_repository_path

        if voter == "arithmetic_mean":
            self.__voter = ArithmeticMeanVoter()
        elif voter == "median":
            self.__voter = MedianVoter()
        else:
            raise Exception(f"Invalid voter: {voter}")

    def evolve(self, evolution_epoch: int, fitness_function_arguments={}) -> None:
        """
        Docstring for evolve

        :param evolution_epoch: Number of generations in the whole evolution process
        :type evolution_epoch: int
        :param: fitness_function_arguments: A dictionary of additional arguments
        - This function must be used by NEATNCLPopulation created by me
        - Arguments required: 
            - model_repository_path (absolute path)
                - str
            - data_train
                - Tuple in the form of (data_train_X: torch.Tensor, data_train_y: torch.Tensor)
            - loss_function
                - Any, e.g. nn.MSELoss
            - min_correlation_penalty_coefficient
                - Float
            - max_correlation_penalty_coefficient
                - Float
            - learning_rate
                - Float
            - ~~max_population_diversity~~ (this will be handled in neat_ncl_population.py instead)
                - Float
        """
        assert self.__population is not None, f"Invalid value of self.__population (expect not None; invoke reset() to restart): {self.__population}"
        assert evolution_epoch > 0, f"Invalid evolution_epoch (expect > 1): {evolution_epoch}"

        # best_genome here does not matter because we are doing ensemble learning

        # Below line is for running with 1 CPU core (I think? )
        self.__best_genome = self.__population.run(
            evaluate_genome, evolution_epoch, fitness_function_arguments=fitness_function_arguments)

        # Below line is for running with all available CPU cores (I think? )
        # We are using a ParallelEvaluator
        # TODO: For now, I do not do optimisation
        # best_genome = self.__population.run(
        #     self.__parallel_evaluator.evaluate, generation_num, fitness_function_arguments=fitness_function_arguments)

        # Get all available genomes
        # NOTE: You can also load from local file system instead
        # TODO: Do not know why the following line does not work -> it returns []
        #     - I think it's because I have not run ncl_update() (not my mistake, it's just my design) -> genomes of new children have no fitness -> return []
        # self.__genomes = self.__reporters.best_genomes(self.__base_learner_num)
        all_genomes = list(self.__population.population.values())

        # Ensure fitness is not None before sorting
        all_genomes = [
            genome for genome in all_genomes if genome.fitness is not None]
        # Sort the current population by fitness and take the top N
        all_genomes.sort(key=lambda g: g.fitness, reverse=True)
        self.__genomes = all_genomes[:self.__base_learner_num]

        # Create all corresponding phenomes
        for genome in self.__genomes:
            phenome = NEATNCLBaseLearner(
                genome=genome,
                config=self.__config,
                genome_id=genome.key,
                model_repository_path=self.__model_repository_path
            )

            self.__phenomes.append(phenome)

    def reset(self) -> None:
        assert self.__config is not None, f"Invalid value of self.__config (expect not None): {self.__config}"

        self.__population = NEATNCLPopulation(self.__config)
        self.__reporters = neat.StatisticsReporter()
        self.__stagnation = neat.DefaultStagnation(
            self.__config.stagnation_config,
            reporters=self.__reporters
        )
        self.__population.reproduction = NEATNCLReproduction(
            self.__config.reproduction_config,
            reporters=self.__reporters,
            stagnation=self.__stagnation
        )
        # Add console output for progress
        # NOTE: Uncomment the line below if you want log
        # self.__population.add_reporter(neat.StdOutReporter(True))

        # For utilising all CPU cores instead of just 1 CPU core
        # num_workers=None will automatically use all available CPU cores (according to AI)
        # TODO: For now, I do not do optimisation
        # self.__parallel_evaluator = neat.ParallelEvaluator(
        #     num_workers=self.__cpu_core_num, eval_function=evaluate_genome)

        self.__genomes = []
        self.__phenomes = []
        self.__best_genome = None

    def load_genomes(self) -> None:
        # TODO: Update
        # NOTE: genome_id (the key in genome.py) might not be consecutive
        # Update self.__genomes
        pass

    def save_genomes(self) -> None:
        # TODO: Update
        pass

    def load_phenomes(self) -> None:
        """
        Load phenomes based on the current genomes
        NOTE: If self.__genomes is empty, no phenomes will be loaded

        :param self: Description
        """
        assert self.__genomes is not None, f"Invalid value of self.__genomes (expect not None -> invoke load_phenomes() first): {self.__genomes}"

        # TODO: Update
        pass

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        """
        A convenient way of invoking model's prediction

        :param: args: Expect contain 1 tensor object representing the input data only
        :return: The output tensor (dimension of the output tensor depends on the problem type)
        """
        assert len(
            args) == 1, f"Expect only passing input-tensor but received invalid argument(s): {args}"
        assert self.__phenomes is not None and len(
            self.__phenomes) > 0, f"No base learners to run prediction (expect at least have 1 base learner): {self.__phenomes}"

        y_prediction_all_raw = []

        for phenome in self.__phenomes:
            y_prediction_base_learner = phenome(args[0])

            y_prediction_all_raw.append(y_prediction_base_learner)

        y_prediction_all = torch.stack(y_prediction_all_raw)

        return self.vote(y_predictions=y_prediction_all)

    def vote(self, y_predictions: torch.Tensor) -> torch.Tensor:
        return self.__voter.vote(y_predictions=y_predictions)

    def train(self) -> None:
        assert self.__phenomes is not None and len(
            self.__phenomes) > 0, f"No base learners to run prediction (expect at least have 1 base learner): {self.__phenomes}"

        for phenome in self.__phenomes:
            phenome.train()

    def eval(self) -> None:
        assert self.__phenomes is not None and len(
            self.__phenomes) > 0, f"No base learners to run prediction (expect at least have 1 base learner): {self.__phenomes}"

        for phenome in self.__phenomes:
            phenome.eval()

    @property
    def max_fitness(self) -> float:
        return self.__max_fitness

    @property
    def base_learner_num(self) -> int:
        return self.__base_learner_num

    @property
    def model_repository_path(self) -> str:
        return self.__model_repository_path

    @property
    def genomes(self) -> list:
        return self.__genomes

    @property
    def phenomes(self) -> list:
        return self.__phenomes

    @property
    def experiment_data(self) -> dict:
        """
        If is_experiment = True, this will be a dict in the form of {generation_index: experiment_data}

        :return: {generation_index: {data_name: data_value}}
        :rtype: dict
        """
        if self.__population is None:
            # Have not created a population to evolve yet
            # Run self.evolve()
            return None

        return self.__population.experiment_data

    @property
    def best_genome(self):
        return self.__best_genome

    @property
    def voter(self) -> Voter:
        return self.__voter
