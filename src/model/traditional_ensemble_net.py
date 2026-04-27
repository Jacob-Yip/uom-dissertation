import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from src.model.mlp import MLP
from src.model.traditional_base_learner import TraditionalBaseLearner
from src.voter.voter_arithmetic_mean import ArithmeticMeanVoter
from src.voter.voter_median import MedianVoter
from src.voter.voter_nn import NNVoter

"""
An ensemble model
"""


class TraditionalEnsembleNet:
    def __init__(self, device_num: int, base_learner_nums: list, model_configurations: dict, dist_backend: str, master_address="localhost", master_port="12355", gloo_socket_filename="lo0", model_repository_path="data/model", voter="arithmetic_mean"):
        """
        :param base_learner_nums: A list of numbers representing number of models running on each GPU; Expect it has length equal to device_num; Order of items in this list matters, going as cuda:0, cuda:1, ..., cuda:(device_num - 1)
        :param voter: The type of voter used
            - arithmetic_mean (default)
            - median
            - nn
        """
        assert device_num >= 1, f"Invalid device_num (expect >= 1): {device_num}"
        assert device_num == len(
            base_learner_nums), f"device_num does not match number of items in base_learner_nums: {device_num} != {len(base_learner_nums)}"

        self.__device_num = device_num
        self.__base_learner_nums = base_learner_nums
        self.__model_configurations = model_configurations
        self.__dist_backend = dist_backend  # nccl for GPU; gloo for CPU
        self.__master_address = master_address
        self.__master_port = master_port
        self.__gloo_socket_filename = gloo_socket_filename  # For running gloo, i.e. CPU
        self.__model_repository_path = model_repository_path

        if voter == "arithmetic_mean":
            self.__voter = ArithmeticMeanVoter()
        elif voter == "median":
            self.__voter = MedianVoter()
        elif voter == "nn":
            self.__voter = NNVoter(ensemble_size=sum(self.__base_learner_nums))
        else:
            raise Exception(f"Invalid voter: {voter}")

    def set_up(self, rank: int) -> None:
        """
        :param: rank: rank = gpu_index in this case; Usually, it means process_index (assuming 1 process per GPU)
        """
        os.environ["MASTER_ADDR"] = self.__master_address
        os.environ["MASTER_PORT"] = self.__master_port
        dist.init_process_group(self.__dist_backend,
                                rank=rank, world_size=self.__device_num)
        torch.cuda.set_device(rank)

    def clean_up(self) -> None:
        dist.destroy_process_group()

    def train_ensemble(self, rank: int, data_loader_train: torch.utils.data.DataLoader, epoch_num: int, learning_rate: float, loss_function: any, epoch_num_per_log=10):
        self.set_up(rank=rank)
        device = torch.device(f"cuda:{rank}")

        # Number of base learner in this GPU
        base_learner_num = self.__base_learner_nums[rank]
        # Create base_learner_num models in this GPU
        base_learners = []
        for device_index in range(base_learner_num):
            model = MLP.build_from_config(
                input_size=self.__model_configurations["input_size"],
                hidden_sizes=self.__model_configurations["hidden_sizes"],
                activations=self.__model_configurations["activations"],
                activation_type=self.__model_configurations["activation_type"],
                output_size=self.__model_configurations["output_size"],
                dropout_rate=self.__model_configurations["dropout_rate"],
                dropout_indices=self.__model_configurations["dropout_indices"]
            )
            base_learner = TraditionalBaseLearner.build_from_config(
                model=model,
                device=device,
                rank=rank,
                device_index=device_index,
                optimizer=torch.optim.Adam(list(model.parameters()) + self.__voter.parameters(), lr=learning_rate),
                loss_function=loss_function,
                use_cpu=False,  # TODO: Hardcode now
                model_repository_path=self.__model_repository_path
            )

            base_learners.append(base_learner)

        for epoch in range(epoch_num):
            batch_losses = []

            for batch_X, batch_y in data_loader_train:
                # .to(device) returns a copy of tensor as it goes from CPU to GPU, i.e. different device
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)

                y_predictions_raw = [None] * base_learner_num

                # Run forward passes concurrently using CUDA streams
                for i in range(base_learner_num):
                    y_predictions_raw[i] = base_learners[i](batch_X)
                    # predictions_raw[i] will be in GPU instead of CPU

                # Synchronise all streams before continuing
                # torch.cuda will point to the right cuda because of torch.cuda.set_device(rank) in set_up()
                torch.cuda.synchronize()

                # Get all local predictions
                # If all items of local_y_predictions are detached, local_y_predictions is also detached
                # However, in this example, we will not be using .detach() because we want to calculate gradients of loss with respect to y_prediction_raw during backpropagation
                """
                NOTE: 
                If y_predictions_raw[i].detach() is changed to y_predictions_raw[i], I will not get any error
                """
                # shape: [base_learner_num, batch_num, 1]
                local_y_predictions = torch.stack([y_prediction_raw.detach(
                ) for y_prediction_raw in y_predictions_raw], dim=0).contiguous()

                # Get all global predictions
                # TODO: Currently, it is assumed that all GPUs have the same number of base learners
                global_y_predictions_raw = [torch.zeros_like(
                    local_y_predictions) for _ in range(self.__device_num)]
                dist.all_gather(global_y_predictions_raw, local_y_predictions)
                # shape: [device_num * base_learner_num, batch_num, 1]
                global_y_predictions = torch.cat(
                    global_y_predictions_raw, dim=0)
                # Calculaet mean prediction: shape [batch_num, 1]
                y_prediction_mean = global_y_predictions.mean(dim=0)

                for i in range(len(base_learners)):
                    # I will not run backpropagation concurrently to avoid tricky bugs
                    # Make sure all tensors passed are in GPU instead of CPU
                    """
                    NOTE: 
                    If y_predictions_raw[i] is changed to local_y_predictions[i], I will get the following error: 
                    element 0 of tensors does not required grad and does not have a grad_fn
                    Reason: 
                    - local_y_predictions is detached as each item in it is generated by y_prediction_raw.detached()
                    - loss_function(local_y_predictions[i], batch_y) in base_learner.train_base_learner() will then be invoked with .backward()
                    - Missing gradient of local_y_predictions[i] for computing backpropagation
                    - Seeing the error
                    """
                    # base_learners[i].train_base_learner(...) will set model to train mode
                    base_learners[i].train_base_learner(
                        y_prediction=y_predictions_raw[i],
                        y_true=batch_y,
                        # TODO: Update to create checkpoints of models
                        save_model=(epoch == epoch_num - 1)
                    )

                # For logging
                batch_losses.append(loss_function(y_prediction_mean, batch_y))

            # Log
            # if rank == 0 and epoch % epoch_num_per_log == 0:
            #     ensemble_loss = torch.mean(torch.tensor(batch_losses))
            #     print(
            #         f"[Epoch {epoch + 1}] Ensemble loss: {ensemble_loss:.4f}")

        self.clean_up()

    def train(self, data_loader_train: torch.utils.data.DataLoader, epoch_num: int, learning_rate: float, loss_function: any, epoch_num_per_log=10) -> None:
        """
        Train the ensemble model
        """
        # join = True is set so that parent process is blocked by child processes, each representing a base learner
        # If set to False, we have to manage child processes manually
        mp.spawn(self.train_ensemble, args=(
            data_loader_train, epoch_num, learning_rate, loss_function, epoch_num_per_log), nprocs=self.__device_num, join=True)

    def evaluate_ensemble(self, rank: int, X: torch.Tensor, queue_y) -> None:
        """
        If using IPC, e.g. mp.Manager().Queue() (by the way, mp.Queue() will not work), models can only be run on 1 machine
        If using DDP, e.g. dist.all_gather(), models can be run on multiple machines
        """
        # TODO: Have not implemented
        # Expect caller to have already invoked self.set_up(rank=rank)
        self.set_up(rank=rank)

        device = torch.device(f"cuda:{rank}")

        # Number of base learner in this GPU
        base_learner_num = self.__base_learner_nums[rank]
        # Create base_learner_num models in this GPU
        base_learners = []
        for device_index in range(base_learner_num):
            base_learner = TraditionalBaseLearner.build_from_config(
                model=None,
                device=device,
                rank=rank,
                device_index=device_index,
                optimizer=None,
                loss_function=None,
                use_cpu=False,  # TODO: Hardcode now
                model_repository_path=self.__model_repository_path
            )
            base_learner.load_state_dict(torch.load(
                f"{base_learner.model_repository_path}/base_learner_traditional_{rank}_{device_index}.pt"), model_configurations=self.__model_configurations)

            base_learners.append(base_learner)

        # .to(device) returns a copy of tensor as it goes from CPU to GPU, i.e. different device
        X = X.to(device)

        y_predictions_raw = [None] * base_learner_num

        # Run forward passes concurrently using CUDA streams
        with torch.no_grad():
            for i in range(base_learner_num):
                base_learners[i].eval()
                y_predictions_raw[i] = base_learners[i](X)
                # predictions_raw[i] will be in GPU instead of CPU

        # Synchronise all streams before continuing
        # torch.cuda will point to the right cuda because of torch.cuda.set_device(rank) in set_up()
        torch.cuda.synchronize()

        # Get all local predictions: shape [base_learner_num, batch_num, 1]
        # If all items of local_y_predictions are detached, local_y_predictions is also detached
        local_y_predictions = torch.stack([y_prediction_raw.detach(
        ) for y_prediction_raw in y_predictions_raw], dim=0).contiguous()

        # Get all global predictions, i.e. global_y_predictions: shape [device_num, base_learner_num, batch_num, 1]
        # TODO: Currently, it is assumed that all GPUs have the same number of base learners
        global_y_predictions_raw = [torch.zeros_like(
            local_y_predictions) for _ in range(self.__device_num)]
        dist.all_gather(global_y_predictions_raw, local_y_predictions)
        # shape [device_num, base_learner_num, batch_num, 1]
        global_y_predictions = torch.stack(
            global_y_predictions_raw, dim=0)

        queue_y.put(global_y_predictions.cpu())

        # TODO: Have not implemented
        # Expect caller to invoke self.clean_up() after this method
        self.clean_up()

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        """
        Testing
        Must load model from filesystem as the process of training models is destroyed after training

        TODO: Have not implemented as I am not sure how to handle model instances created during each process group, e.g. whether they will persist lik the process group or not
        Unlike train_ensemble (which we expect to only run once), evaluate_ensemble() might be invoked for multiple times -> we thus also expect evaluate() to be invoked for multiple times
        Thus, caller is expected to invoke self.set_up() and self.clean_up() at the start and end of evaluation sessions

        :param: X: The input tensor
        :return: The tensor representing the ensemble prediction
        """
        mp.set_start_method("spawn", force=True)  # Ensure compatibility
        """
        TODO: 
        The use of queue limits the scability of this application, i.e. this ensemble model can only run on 1 machine (which hopefully has enough GPUs)
        However, I cannot find a scalable way to get the final predictions from evaluate_ensemble() to evaluate()
        The best dist.gather() can do is to save the final predictions to a shared file system and evaluate() get data from the file system
        To completely solve the problem, I might need to replace mp.spawn(...) with torch.distributed.launch(...)
        """
        queue_y = mp.Manager().Queue()

        mp.spawn(self.evaluate_ensemble, args=(X, queue_y),
                 nprocs=self.__device_num, join=True)

        # Collect results
        # shape: [device_num, base_learner_num, batch_num, 1]
        y_predictions = queue_y.get()
        # shape: [device_num * base_learner_num, batch_num, 1]
        y_predictions_for_voting = y_predictions.flatten(0, 1)

        return self.vote(y_predictions=y_predictions_for_voting)

    def evaluate_all(self, X: torch.Tensor) -> torch.Tensor:
        """
        For experiment
        Similar to evaluate() except it returns all predictions from all base learners before voting, which gives the final prediction after processing

        :param: X: The input tensor
        :return: The tensor representing predictions from all base learners
        """
        mp.set_start_method("spawn", force=True)  # Ensure compatibility
        """
        TODO: 
        The use of queue limits the scability of this application, i.e. this ensemble model can only run on 1 machine (which hopefully has enough GPUs)
        However, I cannot find a scalable way to get the final predictions from evaluate_ensemble() to evaluate()
        The best dist.gather() can do is to save the final predictions to a shared file system and evaluate() get data from the file system
        To completely solve the problem, I might need to replace mp.spawn(...) with torch.distributed.launch(...)
        """
        queue_y = mp.Manager().Queue()

        mp.spawn(self.evaluate_ensemble, args=(X, queue_y),
                 nprocs=self.__device_num, join=True)

        # Collect results
        # shape: [device_num, base_learner_num, batch_num, 1]
        y_predictions = queue_y.get()
        # shape: [device_num * base_learner_num, batch_num, 1]
        y_predictions_all = y_predictions.flatten(0, 1)

        return y_predictions_all

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        """
        A convenient way of invoking model's prediction

        :param: args: Expect contain 1 tensor object representing the input data only
        :return: The output tensor (dimension of the output tensor depends on the problem type)
        """
        return self.evaluate(X=args[0])
    
    def vote(self, y_predictions: torch.Tensor) -> torch.Tensor:
        return self.__voter.vote(y_predictions=y_predictions)

    @property
    def model_repository_path(self) -> str:
        return self.__model_repository_path
