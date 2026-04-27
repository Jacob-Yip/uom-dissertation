import neat
import os
import pickle
from src.model.neat_ncl_base_learner import NEATNCLBaseLearner
from src.neat.neat_ncl_genome import NEATNCLGenome
from src.utils import maths_helper
from src.utils.data_container import Table
import torch

"""
A file containing functions for training NEAT with NCL
"""


def get_genome_outputs(genomes: list, config: neat.Config, data_X: torch.Tensor) -> dict:
    """
    Create a table of outputs from each genome

    :param genomes: [(genome_id, genome_instance)]
    :type genomes: list
    :param config: Configuration instance
    :type config: neat.Config
    :param data_X: A 2D tensor in the form of [[batch_1_data_1, batch_1_data_2, ...]]
    :type data_X: tuple
    :return: A dictionary in the form of {genome_id: genome_output}
    :rtype: dict
    """
    outputs = {}

    for genome_id, genome in genomes:
        model = NEATNCLBaseLearner(
            genome=genome,
            config=config,
            genome_id=genome_id
            # We do not care about model_repository_path here
        )
        model.eval()

        with torch.no_grad():
            outputs[genome_id] = model(data_X)

    return outputs


def get_output_distance(genome_1_id: int, genome_2_id: int, genome_outputs: dict) -> float:
    """
    Compute the difference/distance between the outputs of the 2 genomes
    Formula: 
        output_distance = |y_1 - y_2|
    This function is expected to be commutative, i.e. order of input arguments does not matter

    :param genome_1_id: ID of genome 1
    :type genome_1_id: int
    :param genome_2_id: ID of genome 2
    :type genome_2_id: int
    :param genome_outputs: A dictionary in the form of {genome_id: genome_output}
        - genome_output is in the form of a tensor
    :type genome_outputs: dict
    :return: Output distance
    :rtype: float
    """
    genome_1_output = genome_outputs[genome_1_id]
    genome_2_output = genome_outputs[genome_2_id]

    assert isinstance(
        genome_1_output, torch.Tensor), f"Expect genome_1_output to be a PyTorch tensor: {type(genome_1_output)}"
    assert isinstance(
        genome_2_output, torch.Tensor), f"Expect genome_2_output to be a PyTorch tensor: {type(genome_2_output)}"

    return torch.mean(abs(genome_1_output - genome_2_output), dim=0)


def get_table_genetic_distance(genomes: list, config: neat.Config) -> Table:
    """
    Return table of genetic_distance

    :param genomes: [(genome_id, genome_instance)]
    :type genomes: list
    :param config: The configuration instance
    :type config: neat.Config
    :return: Table of genetic_distance
    :rtype: Table
    """
    table_genetic_distance = Table()

    for genome_1_id, genome_1 in genomes:
        for genome_2_id, genome_2 in genomes:
            if genome_1_id == genome_2_id:
                continue

            table_genetic_distance[(genome_1_id, genome_2_id)] = maths_helper.get_genetic_distance(
                genome_1=genome_1,
                genome_2=genome_2,
                config=config
            )

    return table_genetic_distance


def get_tables_genetic_distance_output_distance_fitness(genomes: list, config: neat.Config, data_X: torch.Tensor) -> tuple:
    """
    Create tables of all combinations of genomes
    Return tables of (genetic_distance, output_distance, fitness)

    :param genomes: [(genome_id, genome_instance)]
    :param config: The configuration instance
    :param data_X: A 2D tensor in the form of [[batch_1_data_1, batch_1_data_2, ...]]
    :type data_X: tuple
    :return: Tables of (genetic_distance, output_distance, fitness)
    """
    table_genetic_distance = Table()
    table_output_distance = Table()
    table_fitness = Table()

    dict_genome_outputs = get_genome_outputs(
        genomes=genomes,
        config=config,
        data_X=data_X
    )

    for genome_1_id, genome_1 in genomes:
        for genome_2_id, genome_2 in genomes:
            # Table of fitness
            if not table_fitness.has_key(key=(genome_1_id, genome_1_id)):
                # Prevent duplicate update
                table_fitness[(genome_1_id, genome_1_id)] = genome_1.fitness

                if genome_1.fitness is None:
                    # First generation: NCL update is not run yet -> no fitness for genome_1
                    table_fitness[(genome_1_id, genome_1_id)] = 0

            if genome_1_id == genome_2_id:
                continue

            # Table of fitness
            if not table_fitness.has_key(key=(genome_2_id, genome_2_id)):
                # Prevent duplicate update
                table_fitness[(genome_2_id, genome_2_id)] = genome_2.fitness

                if genome_2.fitness is None:
                    # First generation: NCL update is not run yet -> no fitness for genome_2
                    table_fitness[(genome_2_id, genome_2_id)] = 0

            # Table of genetic distance
            table_genetic_distance[(genome_1_id, genome_2_id)] = maths_helper.get_genetic_distance(
                genome_1=genome_1,
                genome_2=genome_2,
                config=config
            )

            # Table of output distance
            # NOTE: We have assumed that there is only 1 output feature -> we can use .item() instead of torch.mean()
            # NOTE: torch.mean(..., dim=0) is used to get the mean output distance across batches
            table_output_distance[(genome_1_id, genome_2_id)] = torch.mean(get_output_distance(
                genome_1_id=genome_1_id,
                genome_2_id=genome_2_id,
                genome_outputs=dict_genome_outputs
            ), dim=0).item()

    return (table_genetic_distance, table_output_distance, table_fitness)


def load_genomes(original_genomes: list, model_repository_path: str) -> list:
    """
    Load the genomes stored in the local file system and return the new genome list in the form of [(genome_id, genome_instance)]
    NOTE: Technically, we do not have to return a list cause invoking this method also updates original_genomes

    :param original_genomes: The original list of genomes in the form of [(genome_id, genome_instance)]
    :type original_genomes: list
    :param model_repository_path: Absolute path to the model repository in the local file system
    :type model_repository_path: str
    :return: The updated list of genomes in the form of [(genome_id, genome_instance)]
    :rtype: list
    """
    def get_genome_file_path(genome_id: int, model_repository_path: str) -> str:
        """
        NOTE: Make sure this is identical to genome_file_path() in neat_ncl_base_learner.py

        :return: Absolute path to genome file
        """
        assert genome_id is not None and genome_id >= 0, f"Invalid genome_id (expect >= 0): {genome_id}"

        genome_file_path = os.path.join(
            model_repository_path, f"genome_neat_ncl_{genome_id}.pkl")

        return genome_file_path

    for i in range(len(original_genomes)):
        genome_id, original_genome = original_genomes[i]

        with open(get_genome_file_path(genome_id=genome_id, model_repository_path=model_repository_path), f"rb") as file:
            new_genome = pickle.load(file)

            original_genomes[i] = (genome_id, new_genome)

    return original_genomes


def get_genome_node_num(genome: NEATNCLGenome) -> int:
    """
    Return total number of nodes of a genome, i.e. input_node_num + hidden_node_num + output_node_num
    We will use this number instead of hidden_node_num to classify the whole population into subpopulations
    NOTE: If a connection is disabled, we ignore it (according to the research paper, the way to calculate the number of nodes ignores non-existing associated connection in the network -> look for keyword "associated connection does not exist in the network")

    :param genome: The given genome
    :type genome: NEATNCLGenome
    :return: input_node_num + hidden_node_num + output_node_num
    :rtype: int
    """
    # In the form of {node_id: connection_num}
    node_connection_num = {}
    for key, connection in genome.connections.items():
        if connection.enabled:
            if not key[0] in node_connection_num:
                node_connection_num[key[0]] = 0

            if not key[1] in node_connection_num:
                node_connection_num[key[1]] = 0

            node_connection_num[key[0]] += 1
            node_connection_num[key[1]] += 1

    return len(node_connection_num.items())


def get_sharing_factor(base_learner_num: int, target_genome_id: int, genomes: list, table_genetic_distance: Table, alpha=1, experiment_data=None) -> float:
    """
    Calculate the sharing factor of the target genome
    For the steps, refer to the link https://algorithmafternoon.com/niching/fitness_sharing/

    :param base_learner_num: Number of base learners
    :type base_learner_num: int
    :param target_genome_id: ID of the target genome
    :type target_genome_id: int
    :param genomes: List of genomes
    :type genomes: list
    :param table_genetic_distance: A table of genetic distances between genomes
    :type table_genetic_distance: Table
    :param alpha: A scaling factor for the sharing function (typically set to 1)
    :type alpha: float
    :param experiment_data: The dictionary that stores targeted experiment data that are not returned by this method
    :type experiment_data: dict
    :return: Sharing factor of the target genome
    :rtype: float
    """
    genome_sharing_factor = 0

    for genome_id, genome in genomes:
        if target_genome_id == genome_id:
            # Same genome -> ignore
            continue

        # NOTE: Note that this distance metric can be changed to other metric, e.g. genetic distance, output distance
        d = table_genetic_distance[(target_genome_id, genome_id)]

        sharing_radius = maths_helper.get_niche_radius(
            base_learner_num=base_learner_num, table_genetic_distance=table_genetic_distance)
        
        if experiment_data is not None:
            experiment_data["niche_radiuses"].append(sharing_radius)

        if d < sharing_radius:
            sharing_function_value = 1 - (d / sharing_radius) ** alpha
            genome_sharing_factor += sharing_function_value

    return genome_sharing_factor


def ncl_update(genomes: list, config: neat.Config, model_repository_path: str, data_train: tuple, loss_function: any, correlation_penalty_coefficient: float, learning_rate: float) -> list:
    """
    Apply NCL update on all genomes for 1 time
    Return the updated list of genomes
        - Technically, I can just update the original list because lists are mutable in Python but this is dangerous so I will create a new list of genomes instead
    NOTE: Screw it -> I am doing single process thing as the multi-process thing broke my laptop
    NOTE: Will update genome.fitness

    TODO: Consider using a dictionary argument as training_arguments (cause there are many arguments for training right now)

    :param genomes: List of (genome_id, genome_instance)
    :type genomes: list
    :param model_repository_path: Path to folder containing all models
    :type model_repository_path: str
    :param data_train: A batch of training data sample in the form of [(data_X, data_y)]
    :type data_train: tuple
    :param loss_function: A loss function of any type, e.g. nn.MSELoss()
    :param correlation_penalty_coefficient: The dynamic penalty coefficient calculated using the formula in the research paper
    :param learning_rate: The learning rate for the optimiser
    :return: The list of updated genomes
    :rtype: list
    """
    assert len(data_train) == 2 and isinstance(data_train[0], torch.Tensor) and isinstance(
        data_train[1], torch.Tensor), f"Invalid data_train (expect in the form of (data_train_X, data_train_y)): {data_train}"
    assert data_train[0].ndim == 2 and data_train[
        1].ndim == 2, f"Invalid tensor format (expect data_train[0] and data_train[1] to be in the form of [[batch_1_data_1, batch_1_data_2, ...]]): {data_train[0].shape} and {data_train[1].shape}"

    # In the form of (genome_id, genomes)
    updated_genomes = []

    # NOTE: I commented the lines of code below to prevent a bug -> neat-python might do something that contradicts the lines below
    # torch.set_num_threads(1)
    # torch.set_num_interop_threads(1)

    # .to(device) returns a copy of tensor as it goes from CPU to GPU, i.e. different device
    # 1 batch instead of 1 sample
    data_train_X = data_train[0].to(torch.device("cpu"))
    data_train_y = data_train[1].to(torch.device("cpu"))

    # In the form of {genome_id: phenome}
    phenomes = {}
    # In the form of {genome_id: optimizer_instance}
    optimizers = {}
    # In the form of {genome_id: y_prediction}
    y_predictions = {}

    for genome_id, genome in genomes:
        model = NEATNCLBaseLearner(
            genome=genome,
            config=config,
            genome_id=genome_id,
            model_repository_path=model_repository_path
        )
        phenomes[genome_id] = model
        # Handle when model.parameters() is empty, e.g. do not step when no parameters
        model_parameters = list(model.parameters())

        # Assume nothing to be optimised (as the genome has no connections)
        optimizer = None
        if model_parameters is not None and len(model_parameters) > 0:
            # There are connections to be optimised
            optimizer = torch.optim.Adam(model_parameters, lr=learning_rate)
        optimizers[genome_id] = optimizer

        model.train()
        if optimizer is not None:
            optimizer.zero_grad()

        # NOTE: We assume we train this model 1 time for every method-invokation

        y_prediction = model(data_train_X)
        y_predictions[genome_id] = y_prediction

    y_prediction_mean = torch.mean(
        torch.stack(list(y_predictions.values())), dim=0).detach().cpu()

    for genome_id, genome in genomes:
        model = phenomes[genome_id]
        optimizer = optimizers[genome_id]
        y_prediction = y_predictions[genome_id]

        # Compute NCL loss
        # First term of E_i
        simple_loss = loss_function(y_prediction, data_train_y)
        # Second term of E_i
        # torch.mean() is used here to average the value across batches
        penalty = torch.mean(maths_helper.p(
            y_prediction=y_prediction,  # I think this is not detached
            y_prediction_mean=y_prediction_mean  # I think this is detached
        ), dim=0)
        # E_i
        base_learner_loss = simple_loss + correlation_penalty_coefficient * \
            penalty.mean(dim=0)  # Negative sign of penalty is already included

        base_learner_loss.backward()
        # Do not do optimizer.step() if there are no parameters
        if optimizer is not None:
            optimizer.step()

        model.synchronise()

        # Update genome fitness
        # I decide to do this here instead of somewhere else in ncl_update() because this prevents duplicate calculations
        model.genome.fitness = maths_helper.get_genome_fitness(
            genome_loss=simple_loss,
            penalty=penalty.mean(dim=0),
            weight_decay=maths_helper.get_weight_decay(genome=model.genome)
        ).item()

        genome.fitness = model.genome.fitness
        updated_genomes.append((genome_id, genome))

        # Save updated genome with updated weights to local file system
        # NOTE: So far, there is no need to save the genomes because we are not loading them (instead, we return the list of updated genomes in this method) -> disable saving so that I do not have to load genomes again
        # model.save_genome()

    return updated_genomes


def get_ensemble_loss_average_active_hidden_node_num_y_predictions(genomes: list, config: neat.Config, data_test: tuple, loss_function: any) -> tuple:
    """
    Return (ensemble_loss, average_active_hidden_node_num, y_predictions)

    :param genomes: The list of genomes
    :type genomes: list
    :param config: The configuration instance
    :type config: neat.Config
    :param data_test: (data_test_X, data_test_y)
    :type data_test: tuple
    :param loss_function: The loss function
    :type loss_function: any
    :return: (ensemble_loss, average_active_hidden_node_num, y_predictions)
    :rtype: tuple
    """
    assert len(data_test) == 2 and isinstance(data_test[0], torch.Tensor) and isinstance(
        data_test[1], torch.Tensor), f"Invalid data_test (expect in the form of (data_test_X, data_test_y)): {data_test}"
    assert data_test[0].ndim == 2 and data_test[
        1].ndim == 2, f"Invalid tensor format (expect data_test[0] and data_test[1] to be in the form of [[batch_1_data_1, batch_1_data_2, ...]]): {data_test[0].shape} and {data_test[1].shape}"

    # .to(device) returns a copy of tensor as it goes from CPU to GPU, i.e. different device
    # 1 batch instead of 1 sample
    data_test_X = data_test[0].to(torch.device("cpu"))
    data_test_y = data_test[1].to(torch.device("cpu"))

    active_hidden_node_nums = []

    # In the form of {genome_id: y_prediction}
    genome_id_y_predictions = {}

    for genome_id, genome in genomes:
        model = NEATNCLBaseLearner(
            genome=genome,
            config=config,
            genome_id=genome_id,
            model_repository_path=None
        )

        active_hidden_node_nums.append(model.active_hidden_node_num)

        model.eval()

        with torch.no_grad():
            y_prediction = model(data_test_X)
            genome_id_y_predictions[genome_id] = y_prediction

    # Ensemble prediction
    y_predictions = torch.stack(list(genome_id_y_predictions.values()))
    y_prediction_mean = torch.mean(y_predictions, dim=0).detach().cpu()

    # NOTE: Expect to be a tensor of item only
    ensemble_loss = loss_function(y_prediction_mean, data_test_y).item()

    average_active_hidden_node_num = 0
    if len(active_hidden_node_nums) > 0:
        # Prevent division by 0
        average_active_hidden_node_num = sum(
            active_hidden_node_nums) / len(active_hidden_node_nums)

    return (ensemble_loss, average_active_hidden_node_num, y_predictions)


def check_fitness_function_arguments(fitness_function_arguments: dict) -> None:
    """
    Check if fitness_function_arguments has all the essential arguments and whether there are invalid arguments
    NOTE: The method does not return anything but it raises exception if there is a problem in the format of fitness_function_arguments

    - model_repository_path (absolute path)
        - str
        - Mandatory
    - data_train
        - Tuple in the form of (data_train_X: torch.Tensor, data_train_y: torch.Tensor)
        - Mandatory
        - Expect 1 batch only
    - loss_function
        - Any, e.g. nn.MSELoss
        - Mandatory
    - min_correlation_penalty_coefficient
        - Float
        - Mandatory
    - max_correlation_penalty_coefficient
        - Float
        - Mandatory
    - learning_rate
        - Float
        - Mandatory
    - max_population_diversity
        - Float
        - ~~Mandatory~~
        - This argument is a bit special and we will not check whether it exists in the dictionary
            - It must exist to run the code
            - It's just NEATNCLPopulation manually adds this argument to fitness_function_arguments later on
    - is_experiment
        - bool
        - True if we are performing an experiment -> we will then collect the relevant experiment data, which will be returned by this method
        - Mandatory
    - data_test
        - Tuple in the form of (data_train_X: torch.Tensor, data_train_y: torch.Tensor)
        - Required only when is_experiment = True
        - Expect 1 batch only
        - Optional
    - max_population_diversity_experiment
        - Float
        - Required only when is_experiment = True
        - Optional

    :param fitness_function_arguments: In the form of {argument_name: argument_value}
    :type fitness_function_arguments: dict
    :rtype: None
    """
    assert isinstance(fitness_function_arguments,
                      dict), f"Invalid data type for fitness_function_arguments (expect dict): {type(fitness_function_arguments)}"

    mandatory_arguments = [
        "model_repository_path",
        "data_train",
        "loss_function",
        "min_correlation_penalty_coefficient",
        "max_correlation_penalty_coefficient",
        "learning_rate",
        # "max_population_diversity",
        "is_experiment"
    ]

    for argument_name, argument_value in fitness_function_arguments.items():
        if argument_name == "model_repository_path":
            assert isinstance(argument_value, str) and len(
                argument_value) > 0, f"Invalid model_repository_path: {argument_value}"

            try:
                mandatory_arguments.remove("model_repository_path")
            except Exception as e:
                raise Exception(
                    f"Should not happen: removing model_repository_path from {mandatory_arguments}")
        elif argument_name == "data_train":
            assert isinstance(argument_value, tuple) and len(
                argument_value) == 2, f"Invalid data_train: {argument_value}"

            data_train_X = argument_value[0]
            data_train_y = argument_value[1]

            # Check if they only have 1 batch
            assert isinstance(
                data_train_X, torch.Tensor) and data_train_X.ndim == 2, f"Invalid shape of data_train_X (remember to include all data samples): {data_train_X.shape}"
            assert isinstance(
                data_train_y, torch.Tensor) and data_train_y.ndim == 2, f"Invalid shape of data_train_y (remember to include all data samples): {data_train_y.shape}"

            try:
                mandatory_arguments.remove("data_train")
            except Exception as e:
                raise Exception(
                    f"Should not happen: removing data_train from {mandatory_arguments}")
        elif argument_name == "loss_function":
            assert argument_value is not None, f"Invalid loss_function: {argument_value}"

            try:
                mandatory_arguments.remove("loss_function")
            except Exception as e:
                raise Exception(
                    f"Should not happen: removing loss_function from {mandatory_arguments}")
        elif argument_name == "min_correlation_penalty_coefficient":
            assert (isinstance(argument_value, float) or isinstance(argument_value, int)
                    ) and argument_value >= 0, f"Invalid min_correlation_penalty_coefficient: {argument_value}"

            try:
                mandatory_arguments.remove(
                    "min_correlation_penalty_coefficient")
            except Exception as e:
                raise Exception(
                    f"Should not happen: removing min_correlation_penalty_coefficient from {mandatory_arguments}")
        elif argument_name == "max_correlation_penalty_coefficient":
            # TODO: Check min_correlation_penalty_coefficient <= max_correlation_penalty_coefficient
            assert (isinstance(argument_value, float) or isinstance(
                argument_value, int)), f"Invalid max_correlation_penalty_coefficient: {argument_value}"

            try:
                mandatory_arguments.remove(
                    "max_correlation_penalty_coefficient")
            except Exception as e:
                raise Exception(
                    f"Should not happen: removing max_correlation_penalty_coefficient from {mandatory_arguments}")
        elif argument_name == "learning_rate":
            assert (isinstance(
                argument_value, float) or isinstance(argument_value, int)) and argument_value > 0, f"Invalid learning_rate: {argument_value}"

            try:
                mandatory_arguments.remove("learning_rate")
            except Exception as e:
                raise Exception(
                    f"Should not happen: removing learning_rate from {mandatory_arguments}")
        elif argument_name == "max_population_diversity":
            assert (isinstance(argument_value, float) or isinstance(argument_value, int)
                    ) and argument_value >= 0, f"Invalid max_population_diversity: {argument_value}"

            # A special case (refer to method doc for explanation of why the code below is commented)
            # try:
            #     mandatory_arguments.remove("max_population_diversity")
            # except Exception as e:
            #     raise Exception(
            #         f"Should not happen: removing max_population_diversity from {mandatory_arguments}")
        elif argument_name == "is_experiment":
            assert isinstance(
                argument_value, bool), f"Invalid is_experiment: {argument_value}"

            try:
                mandatory_arguments.remove("is_experiment")
            except Exception as e:
                raise Exception(
                    f"Should not happen: removing is_experiment from {mandatory_arguments}")
        elif argument_name == "data_test":
            # Optional -> no need to remove from manadatory_arguments
            assert isinstance(argument_value, tuple) and len(
                argument_value) == 2, f"Invalid data_test: {argument_value}"

            data_test_X = argument_value[0]
            data_test_y = argument_value[1]

            # Check if they only have 1 batch
            assert isinstance(
                data_test_X, torch.Tensor) and data_test_X.ndim == 2, f"Invalid shape of data_test_X (remember to include all data samples): {data_test_X.shape}"
            assert isinstance(
                data_test_y, torch.Tensor) and data_test_y.ndim == 2, f"Invalid shape of data_test_y (remember to include all data samples): {data_test_y.shape}"
        elif argument_name == "max_population_diversity_experiment":
            # Optional -> no need to remove from manadatory_arguments
            assert (isinstance(argument_value, float) or isinstance(argument_value, int)
                    ) and argument_value >= 0, f"Invalid max_population_diversity_experiment: {argument_value}"
        else:
            raise Exception(
                f"Invalid fitness_function_arguments key: {argument_name}")

    assert len(
        mandatory_arguments) == 0, f"Missing required arguments in fitness_function_arguments: {mandatory_arguments}"


def evaluate_genome(genomes: list, config: neat.Config, fitness_function_arguments: dict) -> dict:
    """
    Will update genome.fitness in 1 generation

    NOTE: Expect data_train and data_test passed via fitness_function_arguments to be 1 batch only so that we can train and test on all data

    We collect experiment data here because this shortens the time taken to run the experiment
    NOTE: Such design is not good for produciton

    :param: fitness_function_arguments: A dictionary of additional arguments
        - This function must be used by NEATNCLPopulation created by me
        - Arguments required: 
            - model_repository_path (absolute path)
                - str
                - Mandatory
            - data_train
                - Tuple in the form of (data_train_X: torch.Tensor, data_train_y: torch.Tensor)
                - Mandatory
                - Expect 1 batch only
            - loss_function
                - Any, e.g. nn.MSELoss
                - Mandatory
            - min_correlation_penalty_coefficient
                - Float
                - Mandatory
            - max_correlation_penalty_coefficient
                - Float
                - Mandatory
            - learning_rate
                - Float
                - Mandatory
            - max_population_diversity
                - Float
                - ~~Mandatory~~
                - This argument is a bit special and we will not check whether it exists in the dictionary
                    - It must exist to run the code
                    - It's just NEATNCLPopulation manually adds this argument to fitness_function_arguments later on
            - is_experiment
                - bool
                - True if we are performing an experiment -> we will then collect the relevant experiment data, which will be returned by this method
                - Mandatory
            - data_test
                - Tuple in the form of (data_train_X: torch.Tensor, data_train_y: torch.Tensor)
                - Required only when is_experiment = True
                - Expect 1 batch only
                - Optional
            - max_population_diversity_experiment
                - Float
                - Required only when is_experiment = True
                - Optional
    :type fitness_function_arguments: dict
    :return: A dictionary of function outputs
        - Outputs returned: 
            - Updated max_population_diversity
            - experiment_data
                - A dictionary in the form of {data_name: data_value}
                    - MSE (test)
                        - Key: ensemble_loss
                    - Diversity coefficient
                        - Key: diversity_coefficient
                    - Average number of hidden nodes
                        - Key: average_active_hidden_node_num
                    - Population diversity
                        - Key: population_diversity
                    - Maximum population diversity
                        - Key: max_population_diversity
                    - Correlation penalty coefficient
                        - Key: correlation_penalty_coefficient
                    - Number of subpopulations
                        - Key: subpopulation_num
                    - [The dynamic niche radius]
                        - Key: niche_radiuses
                        - Expect length = genome_num
                    - [Sharing factor]
                        - Key: sharing_factors
                        - Expect length = genome_num
                    - [Raw fitness]
                        - Key: raw_fitnesses
                        - Expect length = genome_num
                    - [Adjusted fitness]
                        - Key: adjusted_fitnesses
                        - Expect length = genome_num
    :rtype: dict
    """
    # In the form of {data_name: data_value}
    # For collecting experiment data
    experiment_data = {
        "niche_radiuses": [], 
        "sharing_factors": [], 
        "raw_fitnesses": [], 
        "adjusted_fitnesses": []
    }

    # Get arguments for ncl_update
    # Assume all arguments are valid and have been checked by check_fitness_function_arguments(...)
    base_learner_num = len(genomes)
    # Absolute path
    model_repository_path = fitness_function_arguments["model_repository_path"]
    data_train = fitness_function_arguments["data_train"]
    loss_function = fitness_function_arguments["loss_function"]
    # Minimum possible value of correlation penalty coefficient (a constant)
    min_correlation_penalty_coefficient = fitness_function_arguments[
        "min_correlation_penalty_coefficient"]
    # Maximum possible value of correlation penalty coefficient (a constant)
    max_correlation_penalty_coefficient = fitness_function_arguments[
        "max_correlation_penalty_coefficient"]
    learning_rate = fitness_function_arguments["learning_rate"]
    # For training
    max_population_diversity = fitness_function_arguments["max_population_diversity"]

    # For experiment
    is_experiment = fitness_function_arguments["is_experiment"]
    # Testing data
    data_test = None
    # For experiment
    max_population_diversity_experiment = None
    if is_experiment:
        # I expect this to be not invoked if is_experiment = False
        data_test = fitness_function_arguments["data_test"]

        max_population_diversity_experiment = fitness_function_arguments[
            "max_population_diversity_experiment"]

    # Split population into subpopulations based on the number of hidden nodes of each genome
    # NOTE: hidden_node_num is proportional to total_node_num
    # In the form of {total_node_num: [(genome_id, genome)]}
    subpopulation_genome_id_genomes = {}

    for genome_id, genome in genomes:
        genome_total_node_num = get_genome_node_num(genome=genome)

        if not genome_total_node_num in subpopulation_genome_id_genomes:
            subpopulation_genome_id_genomes[genome_total_node_num] = []

        subpopulation_genome_id_genomes[genome_total_node_num].append(
            (genome_id, genome))

    # For training
    # In the form of {genome_total_node_num: (subpopulation_genomes, subpopulation_table_genetic_distance, subpopulation_table_output_distance, subpopulation_table_fitness)}
    subpopulation_datasets_train = {}
    # For experiment
    # Enabled if is_experiment = True
    # In the form of {genome_total_node_num: (subpopulation_genomes, subpopulation_table_genetic_distance, subpopulation_table_output_distance, subpopulation_table_fitness)}
    subpopulation_datasets_test = None
    if is_experiment:
        subpopulation_datasets_test = {}

    for subpopulation_key, subpopulation_genomes in subpopulation_genome_id_genomes.items():
        subpopulation_table_genetic_distance_train, subpopulation_table_output_distance_train, subpopulation_table_fitness_train = get_tables_genetic_distance_output_distance_fitness(
            genomes=subpopulation_genomes,
            config=config,
            data_X=data_train[0]
        )

        subpopulation_datasets_train[subpopulation_key] = (
            subpopulation_genomes, subpopulation_table_genetic_distance_train, subpopulation_table_output_distance_train, subpopulation_table_fitness_train)

        # For collecting experiment data if is_experiment = True
        if is_experiment:
            # Just in case
            assert data_test is not None, f"Missing required argument (expect not None): {data_test}"

            subpopulation_table_genetic_distance_test, subpopulation_table_output_distance_test, subpopulation_table_fitness_test = get_tables_genetic_distance_output_distance_fitness(
                genomes=subpopulation_genomes,
                config=config,
                data_X=data_test[0]
            )

            subpopulation_datasets_test[subpopulation_key] = (
                subpopulation_genomes, subpopulation_table_genetic_distance_test, subpopulation_table_output_distance_test, subpopulation_table_fitness_test)

    current_population_diversity = maths_helper.get_population_diversity(
        subpopulation_datasets=subpopulation_datasets_train)

    # Update max_population_diversity
    max_population_diversity = max(
        max_population_diversity, current_population_diversity)

    # Compute dynamic correlation_penalty_coefficient() for NCL update
    # Formula: lambda' = lambda_max - (current_population_diversity / max_population_diversity) * (lambda_max - lambda_min)
    correlation_penalty_coefficient = maths_helper.get_dynamic_correlation_penalty_coefficient(
        min_correlation_penalty_coefficient=min_correlation_penalty_coefficient,
        max_correlation_penalty_coefficient=max_correlation_penalty_coefficient,
        current_population_diversity=current_population_diversity,
        max_population_diversity=max_population_diversity
    )

    # Apply NCL update before calculation of fitness
    # genome fitness will be updated and stored in local file system
    # Step (c) and step (d) in the research paper
    updated_genomes = ncl_update(
        genomes=genomes,
        config=config,
        model_repository_path=model_repository_path,
        data_train=data_train,
        loss_function=loss_function,
        correlation_penalty_coefficient=correlation_penalty_coefficient,
        learning_rate=learning_rate
    )

    # Local file system is updated here as backpropagation has been applied to genomes already
    # Read from local file system to get the updated genomes (with their updated, temporary fitnesses) here
    # NOTE: Deprecated (too many files are created, which is annoying)
    # genomes = load_genomes(
    #     original_genomes=genomes,
    #     model_repository_path=model_repository_path
    # )
    genomes = updated_genomes

    # NOTE: Most parents from the previous generation will be removed (except those excluded due to elitism) -> most base learners in the current NCL will be their offsprings

    # NOTE: Do not use Table.combine(...) here because that will miss the combinations of genomes from different subpopulations
    table_genetic_distance = get_table_genetic_distance(
        genomes=genomes, config=config)

    # Then, calculate adjusted fitness while taking into account dynamical fitness sharing method
    # NOTE: We will update genome.fitness here to take into account dynamical fitness
    # We perform this here instead of neat_reproduction.py because here, we do not need to pass around arguments
    # Step (a) in the research paper
    for genome_id, genome in genomes:
        genome_sharing_factor = get_sharing_factor(
            base_learner_num=base_learner_num,
            target_genome_id=genome_id,
            genomes=genomes,
            table_genetic_distance=table_genetic_distance, 
            experiment_data=experiment_data
        )

        experiment_data["sharing_factors"].append(genome_sharing_factor)
        experiment_data["raw_fitnesses"].append(genome.fitness)

        # Update genome fitness with the formula genome_raw_fitness / genome_sharing_factor
        if genome_sharing_factor != 0:
            # Prevent division of 0
            genome.fitness = genome.fitness / genome_sharing_factor
        
        experiment_data["adjusted_fitnesses"].append(genome.fitness)

    # Collect experiment data if is_experiment = True
    if is_experiment:
        # Ensemble loss, e.g. MSE
        # Average number of hidden nodes
        experiment_ensemble_loss, experiment_average_active_hidden_node_num, experiment_y_predictions = get_ensemble_loss_average_active_hidden_node_num_y_predictions(
            genomes=genomes,
            config=config,
            data_test=data_test,
            loss_function=loss_function
        )
        experiment_data["ensemble_loss"] = experiment_ensemble_loss
        experiment_data["average_active_hidden_node_num"] = experiment_average_active_hidden_node_num

        # Diversity coefficient
        experiment_diversity_coefficient = torch.mean(maths_helper.diversity_coefficient(
            y_predictions=experiment_y_predictions), dim=0).item()
        experiment_data["diversity_coefficient"] = experiment_diversity_coefficient

        # Population diversity
        """
        Reasons why population diversity might be 0: 
        - Fitness of each genome is 0
            - Happen in the first evolution as the default genome fitness is set to 0 initially
        - There is only 1 genome in the subpopulation, i.e. there is only 1 genome that has n hidden nodes
        """
        experiment_population_diversity = maths_helper.get_population_diversity(subpopulation_datasets=subpopulation_datasets_test, experiment_data=experiment_data)
        experiment_data["population_diversity"] = experiment_population_diversity

        max_population_diversity_experiment = max(
            max_population_diversity_experiment, experiment_population_diversity)
        experiment_data["max_population_diversity"] = max_population_diversity_experiment

        # Correlation penalty coefficient
        experiment_correlation_penalty_coefficient = maths_helper.get_dynamic_correlation_penalty_coefficient(
            min_correlation_penalty_coefficient=min_correlation_penalty_coefficient,
            max_correlation_penalty_coefficient=max_correlation_penalty_coefficient,
            current_population_diversity=experiment_population_diversity,
            max_population_diversity=max_population_diversity_experiment
        )
        experiment_data["correlation_penalty_coefficient"] = experiment_correlation_penalty_coefficient

    # Return the necessary values
    fitness_function_outputs = {
        "max_population_diversity": max_population_diversity
    }

    if is_experiment:
        fitness_function_outputs["experiment_data"] = experiment_data
        fitness_function_outputs["max_population_diversity_experiment"] = max_population_diversity_experiment

    return fitness_function_outputs
