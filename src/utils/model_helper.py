import torch
# Model
from src.model.base_model import BaseModel


class MLPHelper:
    def __init__(self, device: any):
        self.__device = device

    @property
    def device(self):
        return self.__device

    def train_model(self, model: BaseModel, epoch_num: int, loss_function, optimizer, data_loader_train, epoch_num_per_log=20) -> tuple:
        """
        Train a model
        Model will be updated so no need to return the trained model instance
        Will change mode of model to training (if one wishes to test model, remember to change its model back to evaluation)
        Return (total_loss, batch_num)

        :param: model: The model instance
        :param: epoch_num: Number of training epoches
        :param: loss_function: The Python function that calculates the loss
        :param: optimizer: The optimizer instance
        :param: data_loader_train: The training data loader
        :param: epoch_num_per_log: Number of epoches to be finished before logging progress once
        :return: (total_loss, batch_num)
        """
        model.train()

        for epoch in range(epoch_num):
            total_loss = 0

            for (batch_X, batch_y) in data_loader_train:
                batch_X = batch_X.to(self.__device)
                batch_y = batch_y.to(self.__device)

                # y_prediction is a PyTorch tensor
                y_prediction = model(batch_X)

                loss = loss_function(y_prediction, batch_y)

                # Backward pass and optimization
                optimizer.zero_grad()  # reset gradients
                loss.backward()        # compute gradients
                optimizer.step()       # update weights

                total_loss += loss.item()

            if epoch % epoch_num_per_log == 0:
                print(
                    f"Epoch: {epoch + 1} / {epoch_num}; Loss: {(total_loss / len(data_loader_train)):.4f}")

            if epoch == epoch_num - 1:
                # Last epoch
                return (total_loss, len(data_loader_train))

    def test_model(self, model: BaseModel, loss_function, data_loader_test) -> tuple:
        """
        Test a model
        Return (total_loss, batch_num)
        To get the average loss per batch, calculate (total_loss / batch_num)

        :param: model: The model instance
        :param: loss_function: The loss function
        :param: data_loader_test: The testing data loader
        :return: (total_loss, batch_num)
        """
        model.eval()

        with torch.no_grad():
            # Tell PyTorch not to compute gradients because it is not needed in evaluation -> save memory by not tracking gradients along calculations
            total_loss = 0

            for (batch_X, batch_y) in data_loader_test:
                batch_X = batch_X.to(self.__device)
                batch_y = batch_y.to(self.__device)

                y_prediction = model(batch_X)

                loss = loss_function(y_prediction, batch_y)

                total_loss += loss.item()

            return (total_loss, len(data_loader_test))
