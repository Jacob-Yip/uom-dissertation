import neat
from src.neat.neat_ncl_genome import NEATNCLGenome
from src.utils.data_container import Table
import torch


def p(y_prediction: torch.Tensor, y_prediction_mean: torch.Tensor) -> torch.Tensor:
    """
    p_i(n) is the correlation penalty function
    p_i(n) = -(f_i(n) - f_bar(n))^2
    For calculation of negative correlation learning
    NOTE: In some research paper, the value is sum but in this project, we calculate the penalty value (torch.mean(...) is applied at caller level)

    :param: y_prediction: Tensor of all predictions of the current base learner. Expect size (batch_size, 1)
    :param: y_prediction_mean: Tensor of mean of predictions of all base learners. Expect size (batch_size, 1)
    :return: -(y_predictions - y_prediction_mean)^2. Expect size (batch_size, 1)
    """
    return -1 * (y_prediction - y_prediction_mean) ** 2


def diversity_coefficient(y_predictions: torch.Tensor) -> torch.Tensor:
    """
    (1 / m) * summation of (-p)

    :param: y_predictions: Tensors of all predictions of the current base learners. Expect size (base_learner_num, batch_size, 1)
    :return: diversity_coefficients. Expect size (batch_size, 1)
    """
    # Shape: [batch_size, 1]
    y_prediction_mean = torch.mean(y_predictions, dim=0)

    # Shape [base_learner_num, batch_size, 1]
    # All mean values are copied
    mean_expanded = y_prediction_mean.unsqueeze(0).expand(
        y_predictions.shape[0], *([-1] * y_prediction_mean.dim()))

    diversity_sum = (y_predictions - mean_expanded) ** 2

    return torch.mean(diversity_sum, dim=0)


def get_niche_radius(base_learner_num: int, table_genetic_distance: Table) -> float:
    """
    Calculate the dynamic niche_radius
    Formula: 
        niche_radius = 1 / (2 * base_learner_num ** 2) * (sum_of_all_combinations_of_genome_distances)

    :param base_learner_num: Number of base learners
    :type base_learner_num: int
    :param genomes: A list of genome instances
    :type genomes: list
    :return: niche radius
    :rtype: float
    """
    return (1 / (2 * base_learner_num ** 2)) * sum(table_genetic_distance.values)


def get_dynamic_correlation_penalty_coefficient(min_correlation_penalty_coefficient: float, max_correlation_penalty_coefficient: float, current_population_diversity: float, max_population_diversity: float) -> float:
    """
    Calculate the value of the dynamic penalty coefficient, i.e. lambda
    Formula: 
        lambda' = lambda_max - (D / D_max) * (lambda_max - lambda_min)

    :param min_correlation_penalty_coefficient: Minimum possible value of penalty coefficient, e.g. 0.1 in the research paper
    :type min_correlation_penalty_coefficient: float
    :param max_correlation_penalty_coefficient: Maximum possible value of penalty coefficient, e.g. 0.9 in the research paper
    :type max_correlation_penalty_coefficient: float
    :param current_population_diversity: Current population diversity
    :type current_population_diversity: float
    :param max_population_diversity: Maximum population diversity recorded
    :type max_population_diversity: float
    :return: The value of the dynamic penalty coefficient, which is within the range [min_correlation_penalty_coefficient, max_correlation_penalty_coefficient]
    :rtype: float
    """
    assert min_correlation_penalty_coefficient >= 0, f"Invalid min_correlation_penalty_coefficient (expect >= 0): {min_correlation_penalty_coefficient}"
    assert max_correlation_penalty_coefficient >= 0, f"Invalid max_correlation_penalty_coefficient (expect >= 0): {max_correlation_penalty_coefficient}"
    assert min_correlation_penalty_coefficient <= max_correlation_penalty_coefficient, f"Invalid min_correlation_penalty_coefficient and max_correlation_penalty_coefficient (expect min_correlation_penalty_coefficient <= max_correlation_penalty_coefficient): {min_correlation_penalty_coefficient} > {max_correlation_penalty_coefficient}"
    assert current_population_diversity <= max_population_diversity, f"Expect current_population_diversity <= max_population_diversity but: {current_population_diversity} > {max_population_diversity}"

    if max_population_diversity == 0:
        # First generation: NCL update is not run yet -> all genomes have fitness = 0
        # Prevent division by zero
        # For the first generation, we maximise the value of correlation penalty coefficient
        return max_correlation_penalty_coefficient
    else:
        return max_correlation_penalty_coefficient - (current_population_diversity / max_population_diversity) * (max_correlation_penalty_coefficient - min_correlation_penalty_coefficient)


def get_weight_decay(genome: NEATNCLGenome) -> float:
    """
    Compute the summation of (w_i_j^2) / (1 + w_i_j^2), where w_i_j is a weight in a genome for all weights

    :param genome: The given genome
    :type genome: NEATNCLGenome
    :return: The weight decay
    :rtype: float
    """
    weight_decay_term = 0.0

    for connection_key, connection in genome.connections.items():
        # connection_key is in the form of (start_node_index, end_node_index)
        # connection is an instance
        if connection.enabled:
            weight_decay_term += (connection.weight ** 2) / \
                (1 + connection.weight ** 2)

    return weight_decay_term


def get_genetic_distance(genome_1: NEATNCLGenome, genome_2: NEATNCLGenome, config: neat.Config) -> float:
    """
    Compute the difference/distance between 2 genomes
    Formula: 
        genetic_distance = c_1 * (excess_gene / larger_genome_size) + c_2 * (disjoint_gene / larger_genome_size) + c3 * average_weight_difference_of_matching_genes
    Matching genes are genes with the same innovation ID
    This function is expected to be commutative, i.e. order of input arguments does not matter

    :param genome_1: The first genome to be compared
    :type genome_1: NEATNCLGenome
    :param genome_2: The second genome to be compared
    :type genome_2: NEATNCLGenome
    :param config: Configuration instance
    :type config: neat.Config
    :return: The distance between the 2 input genomes
    :rtype: float
    """
    assert genome_1 is not None, f"Missing required argument genome_1: {genome_1}"
    assert genome_2 is not None, f"Missing required argument genome_2: {genome_2}"

    return genome_1.distance(other=genome_2, config=config)


def get_average_genome_fitness(genomes: list, table_fitness: Table) -> float:
    """
    Calculate the average fitness value

    :param genomes: [(genome_id, genome_instance)]
    :type genomes: list
    :param table_fitness: The table of fitnesses of all genomes whose keys are (genome_id, genome_id)
    :type table_fitness: Table
    :return: The average fitness value
    :rtype: float
    """
    fitness_value_sum = 0

    for genome_id, genome in genomes:
        fitness_value_sum += table_fitness[(genome_id, genome_id)]

    return fitness_value_sum / len(genomes)


def get_population_diversity(subpopulation_datasets: dict, experiment_data=None) -> float:
    """
    Calculate the population diversity
    NOTE: Each subpopulation is a collection of genomes which have the same number of hidden nodes
    For the steps, refer to the research paper (right now, there are 5 steps)

    :param subpopulation_datasets: The set of all statistics of each subpopulation in the form of {total_node_num: (subpopulation_genomes, subpopulation_1_table_genetic_distance, subpopulation_1_table_output_distance, subpopulation_1_table_fitness)}
    :type subpopulation_datasets: dict
    :param experiment_data: The dictionary for storing useful metrics that are not returned by this method
    :type experiment_data: dict
    :return: The population diversity of the whole population
    :rtype: float
    """
    # In the form of {total_node_num: subpopulation_diversity}
    # NOTE: No need a table actually -> commented to save memory
    # subpopulation_diversities = {}
    population_diversity = 0

    for subpopulation_key, subpopulation_dataset in subpopulation_datasets.items():
        # Calculate subpopulation diversity
        # SD in the research paper
        # NOTE: No need a table actually -> commented to save memory
        # subpopulation_diversity = Table()

        # In the form of [(genome_1_id, genome_2_id)], where genome_1_id <= genome_2_id
        seen_keys = []

        sd_sum = 0

        subpopulation_genomes = subpopulation_dataset[0]
        subpopulation_table_genetic_distance = subpopulation_dataset[1]
        subpopulation_table_output_distance = subpopulation_dataset[2]
        subpopulation_table_fitness = subpopulation_dataset[3]

        subpopulation_average_fitness = get_average_genome_fitness(
            genomes=subpopulation_genomes,
            table_fitness=subpopulation_table_fitness
        )

        for genome_1_id, genome_1 in subpopulation_genomes:
            for genome_2_id, genome_2 in subpopulation_genomes:
                if genome_1_id == genome_2_id:
                    continue

                cell_key = tuple(sorted((genome_1_id, genome_2_id)))

                if cell_key in seen_keys:
                    # Prevent duplicate update
                    continue
                else:
                    seen_keys.append(cell_key)

                # Add 1 cell to the table subpopulation_diversity (if table is enabled)
                # Formula: DS_i_j = (fitness_1 + fitness_2) / (2 * average_fitness) * genetic_distance * output_distance
                # The term DS_i_j in the research paper
                if subpopulation_average_fitness == 0:
                    # First generation: NCL update is not run yet -> all genomes have fitness = 0
                    # This is to prevent division by 0
                    # Since all genomes have fitness = 0, subpopulation diversity = 0
                    sd_i_j = 0
                else:
                    sd_i_j = ((subpopulation_table_fitness[(genome_1_id, genome_1_id)] + subpopulation_table_fitness[(genome_2_id, genome_2_id)]) / (
                        2 * subpopulation_average_fitness)) * subpopulation_table_genetic_distance[cell_key] * subpopulation_table_output_distance[cell_key]

                sd_sum += sd_i_j

        # From the research paper, k is (number of genomes - 1)
        k = len(subpopulation_genomes) - 1
        # Formula: SD = 2 / (k * (k - 1)) * summation of sd_i_j
        if k * (k - 1) == 0:
            # k = 1
            # Only 1 genome in that subpopulation
            # Prevent division by 0
            subpopulation_diversity = sd_sum
        else:
            subpopulation_diversity = (2 / (k * (k - 1))) * sd_sum

        population_diversity += subpopulation_diversity
    
    # Update experiment data
    if experiment_data is not None:
        experiment_data["subpopulation_num"] = len(subpopulation_datasets)

    return population_diversity


def get_genome_fitness(genome_loss: torch.Tensor, penalty: torch.Tensor, weight_decay: float) -> torch.Tensor:
    """
    The function to calculate the fitness of a genome
    Formula: 
        genome_fitness = -(loss + penalty (with a negative sign) + weight_decay)
        NOTE: We multiply it by -1 because in the research paper, they are trying to minimise fitness but in neat-python we should be maximising fitness

    :param genome_loss: The loss of this genome, e.g. MSE Loss, in the form of a 1D tensor
    :type genome_loss: torch.Tensor
    :param penalty: Second term of the equation in the form of a 1D tensor
    :type penalty: torch.Tensor
    :param weight_decay: Weight decay
    :type weight_decay: float
    :return: The fitness of genome in the form of a 1D tensor
    :rtype: torch.Tensor
    """
    assert isinstance(genome_loss, torch.Tensor) and isinstance(penalty, torch.Tensor) and genome_loss.dtype == penalty.dtype and isinstance(
        weight_decay, float), f"Invalid types of arguments (expect to be torch.tensor with dtype d, torch.tensor with dtype d, float) but received: {type(genome_loss)}, {type(penalty)}, {type(weight_decay)}"
    assert genome_loss.shape == penalty.shape, f"Mismatch between shapes of genome_loss and penalty: {genome_loss.shape} != {penalty.shape}"

    return -1 * (genome_loss + penalty + torch.tensor(weight_decay, dtype=genome_loss.dtype))


def average(data: list) -> float:
    """
    Calculate the average value of a list of data

    :param data: The list of data
    :type data: list
    :return: The average value of data
    :rtype: float
    """
    return sum(data) / len(data)
