import neat
import torch
from src.model.traditional_base_learner import TraditionalBaseLearner
from src.model.mlp import MLP
from src.neat.neat_ncl_genome import NEATNCLGenome

"""
A wrapper for any model, e.g. mlp
Contains functions, e.g. training
Base learner for a NEAT model
Note that this base learner is actually more like a population
    - Use __model to access the best representative model
"""


class NEATBaseLearner:
    def __init__(self, config: neat.Config, evolution_epoch: int, max_fitness: float, device, rank: int, device_index: int, loss_function, use_cpu=False, model_repository_path="data/model"):
        """
        Create a base learner for traditional ensemble learner
        world_size = device_num * model_num_per_gpu

        :param: rank: Unique ID of the GPU running this base learner among the ensemble model; Start from 0 inclusively; Expect value [0, device_num)
        :param: device_index: Unique local ID of this base learner in this GPU; Final unique ID is rank_device_index; Start from 0 inclusively
        :param: use_cpu: True if we are training the model with a CPU
        """
        assert not use_cpu, f"We do not support CPU currently"

        # NEAT metadata
        assert not config is None, f"Missing required parameter config: {config}"
        self.__config = config
        self.__population = neat.Population(self.__config)
        # For statistics purposes, e.g. getting the best model
        self.__reporter_statistics = neat.StatisticsReporter()
        self.__population.add_reporter(self.__reporter_statistics)
        assert evolution_epoch > 0, f"Invalid epochs for evolution: {evolution_epoch}"
        self.__evolution_epoch = evolution_epoch
        self.__max_fitness = max_fitness

        # Model metadata
        # The best model in the current generation
        self.__model = None
        self.__model_genome = None  # Genome of self.__model
        self.__device = device
        self.__rank = rank
        self.__device_index = device_index

        if not self.__model is None and not self.__model.model is None:
            # If self.__model or self.__model.modele is None, that means it has not been loaded
            # E.g. during evaluation instead of training
            self.__model.model.to(self.__device)  # Move device to GPU

        # For training
        self.__loss_function = loss_function

        # Other configuration data
        # TODO: Will not be used now
        self.__use_cpu = use_cpu
        self.__model_repository_path = model_repository_path

    @classmethod
    def build_from_config(cls, config: neat.Config, evolution_epoch: int, max_fitness: float, device, rank: int, device_index: int, loss_function, use_cpu=False, model_repository_path="data/model"):
        return NEATBaseLearner(
            config=config,
            evolution_epoch=evolution_epoch,
            max_fitness=max_fitness,
            device=device,
            rank=rank,
            device_index=device_index,
            loss_function=loss_function,
            use_cpu=use_cpu,
            model_repository_path=model_repository_path
        )

    @classmethod
    def build_from_config_file(cls, config_path: str, evolution_epoch: int, max_fitness: float, device, rank: int, device_index: int, loss_function, use_cpu=False, model_repository_path="data/model"):
        config = neat.Config(NEATNCLGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

        return NEATBaseLearner(
            config=config,
            evolution_epoch=evolution_epoch,
            max_fitness=max_fitness,
            device=device,
            rank=rank,
            device_index=device_index,
            loss_function=loss_function,
            use_cpu=use_cpu,
            model_repository_path=model_repository_path
        )

    @property
    def config(self) -> neat.Config:
        return self.__config

    @property
    def population(self) -> neat.Population:
        return self.__population

    @property
    def reporter_statistics(self) -> neat.StatisticsReporter:
        return self.__reporter_statistics

    @property
    def evolution_epoch(self) -> int:
        return self.__evolution_epoch

    @property
    def max_fitness(self) -> float:
        return self.__max_fitness

    @property
    def device(self) -> torch.device:
        return self.__device

    @property
    def rank(self) -> int:
        return self.__rank

    @property
    def device_index(self) -> int:
        return self.__device_index

    @property
    def model_repository_path(self) -> str:
        return self.__model_repository_path

    @property
    def best_genomes(self, n: int) -> list:
        return self.__reporter_statistics.best_genomes(n=n)

    @property
    def layer_heights(self) -> list:
        return self.__model.layer_heights

    @property
    def model(self) -> TraditionalBaseLearner:
        return self.__model

    @property
    def model_genome(self) -> NEATNCLGenome:
        return self.__model_genome

    @property
    def genome_nodes(self) -> dict:
        """
        NOTE: Does not take into account input neurons
            - For code consistency
                - We will not return disabled connection from input neurons to other neurons
                    - Does not make sense to show edges with weights 0
                - Thus, we should not show input neurons that do not contribute to the final predictions

        :return: {node_id, neat_node_instance}
        """
        if self.__model_genome is None:
            # Best genome does not exist yet
            # Need to evolve(...) first
            return None
        else:
            return self.__model_genome.nodes

    @property
    def genome_connections(self) -> dict:
        """
        :return: {connection_key: neat_connection_instance}
        """
        if self.__model_genome is None:
            # Best genome does not exist yet
            # Need to evolve(...) first
            return None
        else:
            return self.__model_genome.connections

    def evolve(self, data_loader_train: torch.utils.data.DataLoader, train_epoch=None, learning_rate=None) -> TraditionalBaseLearner:
        """
        Evolve the population represented by this class
        Will update
        - self.__model
        - self.__model_genome
        """
        assert not self.__population is None, f"Missing population for evolution: {self.__population}"

        self.__model_genome = self.__population.run(
            self.make_evaluate_genomes(
                data_loader_train=data_loader_train,
                train_epoch=train_epoch,
                learning_rate=learning_rate
            ),
            self.__evolution_epoch
        )

        # This will move the model to GPU
        best_model = TraditionalBaseLearner.build_from_genome(
            genome=self.__model_genome,
            config=self.__config,
            device=self.__device,
            rank=self.__rank,
            device_index=self.__device_index,
            learning_rate=learning_rate,
            loss_function=self.__loss_function,
            use_cpu=self.__use_cpu,
            model_repository_path=self.__model_repository_path
        )

        if not train_epoch is None:
            # Train this model
            assert not learning_rate is None and learning_rate > 0, f"Invalid argument learning_rate for training selected model: {learning_rate}"

            optimizer_best_model = torch.optim.Adam(
                best_model.model.parameters(), lr=learning_rate)

            # We will apply back propagation to genome before selection
            for _ in range(train_epoch):
                for batch_X, batch_y in data_loader_train:
                    # .to(device) returns a copy of tensor as it goes from CPU to GPU, i.e. different device
                    batch_X = batch_X.to(self.__device)
                    batch_y = batch_y.to(self.__device)

                    y_prediction = best_model(batch_X)

                    # Synchronise all streams before continuing
                    # torch.cuda will point to the right cuda because of torch.cuda.set_device(rank) in set_up()
                    # NOTE: Should not affect anything here because we only have 1 base learner at the moment
                    torch.cuda.synchronize()

                    optimizer_best_model.zero_grad()

                    best_model_loss = self.__loss_function(
                        y_prediction, batch_y)
                    best_model_loss.backward()

                    optimizer_best_model.step()

        self.__model = best_model

    def make_evaluate_genomes(self, data_loader_train: torch.utils.data.DataLoader, train_epoch=None, learning_rate=None) -> any:
        """
        This function exists because evaluate_genomes(...) must only have arguments genomes and config
        """

        def evaluate_genomes(genomes, config) -> None:
            """
            Fitness evaluation

            NOTE: This trains each genome sequentially on a GPU
                - Not the most efficient way but if I want to optimise it, I need to learn a lot more so won't be optimising it now

            :param: train_epoch: Number of epochs to train the model before next generation
            """
            for genome_id, genome in genomes:
                # This will move the model to GPU
                net = TraditionalBaseLearner.build_from_genome(
                    genome=genome,
                    config=config,
                    device=self.__device,
                    rank=self.__rank,
                    device_index=self.__device_index,
                    learning_rate=learning_rate,
                    loss_function=self.__loss_function,
                    use_cpu=self.__use_cpu,
                    model_repository_path=self.__model_repository_path
                )

                # Set to training mode
                net.train()

                fitness = self.__max_fitness  # Maximum possible fitness
                batch_losses = []
                if train_epoch is None:
                    # We just use the weights from genome directly
                    for batch_X, batch_y in data_loader_train:
                        # .to(device) returns a copy of tensor as it goes from CPU to GPU, i.e. different device
                        batch_X = batch_X.to(self.__device)
                        batch_y = batch_y.to(self.__device)

                        y_prediction = net(batch_X)

                        # Synchronise all streams before continuing
                        # torch.cuda will point to the right cuda because of torch.cuda.set_device(rank) in set_up()
                        # NOTE: Should not affect anything here because we only have 1 base learner at the moment
                        torch.cuda.synchronize()

                        base_learner_loss = self.__loss_function(
                            y_prediction, batch_y)
                        batch_losses.append(base_learner_loss)
                else:
                    assert not learning_rate is None and learning_rate > 0, f"Invalid argument learning_rate for training unselected model: {learning_rate}"

                    optimizer = net.optimizer

                    # We will apply back propagation to genome before selection
                    for _ in range(train_epoch):
                        for batch_X, batch_y in data_loader_train:
                            # .to(device) returns a copy of tensor as it goes from CPU to GPU, i.e. different device
                            batch_X = batch_X.to(self.__device)
                            batch_y = batch_y.to(self.__device)

                            y_prediction = net(batch_X)

                            # Synchronise all streams before continuing
                            # torch.cuda will point to the right cuda because of torch.cuda.set_device(rank) in set_up()
                            # NOTE: Should not affect anything here because we only have 1 base learner at the moment
                            torch.cuda.synchronize()

                            optimizer.zero_grad()

                            base_learner_loss = self.__loss_function(
                                y_prediction, batch_y)
                            base_learner_loss.backward()
                            batch_losses.append(base_learner_loss)

                            optimizer.step()

                average_batch_loss = torch.mean(
                    torch.tensor(batch_losses)).item()

                # TODO: Need to select a good maximum fitness value
                fitness -= average_batch_loss

                genome.fitness = fitness

        return evaluate_genomes

    def __call__(self, *args, **kwargs) -> any:
        """
        Wrapper function of PyTorch model
        with torch.cuda.stream(self.__stream) is wrapped around method
        Expect all tensors have been moved to the device in caller
        NOTE: Caller does not need to manually set mode, i.e. train/eval, nor using torch.no_grad()

        :return: prediction-tensor of model
        """
        with torch.no_grad():
            self.eval()

            return self.__model(*args, **kwargs)

    def eval(self) -> None:
        """
        Wrapper function of PyTorch model
        """
        self.__model.eval()

    def train(self) -> None:
        """
        Wrapper function of PyTorch model
        """
        self.__model.train()

    def load_state_dict(self, state_dict, strict=True, assign=False, **kwargs) -> None:
        """
        Wrapper function of PyTorch model

        :param: kwargs: Expect {"model_configurations": dictionary of model configurations} if self.__model is None
        """
        if self.__model.model is None:
            assert "model_configurations" in kwargs, f"Missing required model configurations to create model: {kwargs}"

            model_configurations = kwargs["model_configurations"]

            self.__model.model = MLP.build_from_config(
                input_size=model_configurations["input_size"],
                hidden_sizes=model_configurations["hidden_sizes"],
                activations=model_configurations["activations"],
                activation_type=model_configurations["activation_type"],
                output_size=model_configurations["output_size"],
                dropout_rate=model_configurations["dropout_rate"],
                dropout_indices=model_configurations["dropout_indices"]
            )

        self.__model.model.load_state_dict(
            state_dict=state_dict, strict=strict, assign=assign)

        # Move device to the correct GPU
        assert not self.__device is None, f"Expect device set during instantiation: {self.__device}"
        self.__model.model.to(self.__device)

    def save_model(self) -> None:
        """
        Save model at self.__model_repository_path
        """
        torch.save(self.__model.model.state_dict(),
                   f"{self.__model_repository_path}/base_learner_neat_{self.__rank}_{self.__device_index}.pt")
