import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import os
from torch.nn.parallel import DistributedDataParallel as DDP

# ======================
# Simple MLP model
# ======================


class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(13, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

# ======================
# Initialize process group
# ======================


def setup(rank, world_size, master_address="localhost", master_port="12235"):
    os.environ["MASTER_ADDR"] = master_address
    os.environ["MASTER_PORT"] = master_port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()

# ======================
# Training function
# ======================


def train(rank, world_size, epochs=10, epoch_num_per_log=2, batch_size=32, lambda_ncl=0.5):
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    # Dummy dataset (simulate Boston housing)
    N = 64
    X = torch.randn(N, 13).to(device)
    y = torch.randn(N, 1).to(device)

    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True)

    # Create 4 models per GPU
    model_num_per_gpu = 4
    models = [SimpleMLP().to(device) for _ in range(model_num_per_gpu)]
    optimizers = [torch.optim.Adam(model.parameters(), lr=0.01)
                  for model in models]
    loss_fn = nn.MSELoss()

    # Create separate CUDA streams for parallel forward passes
    streams = [torch.cuda.Stream(device) for _ in range(model_num_per_gpu)]

    for epoch in range(epochs):
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            predictions_raw = [None] * model_num_per_gpu

            # Run forward passes concurrently using CUDA streams
            for i in range(model_num_per_gpu):
                with torch.cuda.stream(streams[i]):
                    predictions_raw[i] = models[i](batch_X)

            # Synchronize all streams before continuing
            torch.cuda.synchronize()

            # Concatenate local predictions: shape [4 * B, 1]
            local_predictions = torch.cat([prediction_raw.detach(
            ) for prediction_raw in predictions_raw], dim=0).contiguous()

            # All-gather across all processes
            global_model_num = world_size * model_num_per_gpu
            # Might not be equal to batch_size for the last batch
            batch_num = batch_X.shape[0]

            global_predictions_raw = [torch.zeros_like(local_predictions)
                                      for _ in range(world_size)]
            dist.all_gather(global_predictions_raw, local_predictions)

            global_predictions = torch.cat(global_predictions_raw, dim=0).view(
                global_model_num, batch_num, -1)
            mean_prediction = global_predictions.mean(dim=0)  # shape: [B, 1]

            for i in range(model_num_per_gpu):
                optimizers[i].zero_grad()
                mse_loss = loss_fn(predictions_raw[i], batch_y)
                ncl_penalty = torch.mean(
                    (predictions_raw[i].detach() - mean_prediction.to(device)) ** 2)
                print(f"mse_loss shape: {mse_loss.shape}")  # scalar
                print(f"ncl_penalty shape: {ncl_penalty.shape}")  # scalar
                total_loss = mse_loss + lambda_ncl * ncl_penalty
                total_loss.backward()
                optimizers[i].step()

            if rank == 0 and epoch % epoch_num_per_log == 0:
                print(
                    f"[Epoch {epoch}] Losses (model 0-3): {[loss_fn(p, batch_y).item() for p in predictions_raw]}")
                print(f"local_predictions: {local_predictions.shape}")  # [256, 1]
                print(f"global_predictions: {global_predictions.shape}")  # [8, 32, 1]
                print(f"mean_prediction: {mean_prediction.shape}")  # [32, 1]

    cleanup()

# ======================
# Entry point for multi-process training
# ======================


def main():
    device_num = torch.cuda.device_count()
    mp.spawn(train, args=(device_num,), nprocs=device_num, join=True)


if __name__ == "__main__":
    main()
