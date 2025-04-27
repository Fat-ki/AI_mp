# neuralnet.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/29/2019
# Modified by James Soole for the Fall 2023 semester

"""
This is the main entry point for MP10 Part2. You should only modify code within this file.
The unrevised staff files will be used for all other files and classes when code is run, 
so be careful to not modify anything else.
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import get_dataset_from_arrays
from torch.utils.data import DataLoader


class NeuralNet(nn.Module):
    def __init__(self, lrate, loss_fn, in_size, out_size):
        """
        Initializes the layers of your neural network.

        Parameters:
        lrate (float): Learning rate for the model.
        loss_fn (callable): A loss function defined as follows:
            Parameters:
                yhat (Tensor): An (N, out_size) Tensor.
                y (Tensor): An (N,) Tensor.
            Returns:
                Tensor: A scalar Tensor that is the mean loss.
        in_size (int): Input dimension.
        out_size (int): Output dimension.
        """
        super(NeuralNet, self).__init__()
        self.loss_fn = loss_fn

        # For Part 1, the network should have the following architecture (in terms of hidden units):
        # in_size -> h -> out_size, where 1 <= h <= 256

        # TODO Define the network architecture (layers) based on these specifications.
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_size, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(1152, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, out_size)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=lrate)

    def forward(self, x):
        """
        Performs a forward pass through your neural net (evaluates f(x)).

        Parameters:
        x (Tensor): An (N, in_size) Tensor.

        Returns:
        Tensor: An (N, out_size) Tensor of output from the network.
        """
        # TODO Implement the forward pass.
        x = x.view(-1, 3, 31, 31)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

    def step(self, x, y):
        """
        Performs one gradient step through a batch of data x with labels y.

        Parameters:
        x (Tensor): An (N, in_size) Tensor representing the input data.
        y (Tensor): An (N,) Tensor representing the labels.

        Returns:
        float: The total empirical risk (mean of losses) for this batch.
        """
        yhat = self.forward(x)
        loss = self.loss_fn(yhat, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()


def fit(train_set,train_labels,dev_set,epochs,batch_size=100):
    """
    Creates and trains a NeuralNet object 'net'. Use net.step() to train the neural net
    and net(x) to evaluate the neural net.

    Parameters:
    train_set (Tensor): An (N, in_size) Tensor representing the training data.
    train_labels (Tensor): An (N,) Tensor representing the training labels.
    dev_set (Tensor): An (M,) Tensor representing the development set.
    epochs (int): The number of training epochs.
    batch_size (int, optional): The size of each training batch. Defaults to 100.

    This method must work for arbitrary M and N.

    The model's performance could be sensitive to the choice of learning rate.
    We recommend trying different values if your initial choice does not work well.
    For Part 1, we recommend setting the learning rate to 0.01.

    Returns:
    list: A list of floats containing the total loss for every epoch.
        Ensure that len(losses) == epochs.
    numpy.ndarray: An (M,) NumPy array (dtype=np.int64) of estimated class labels (0,1,2, or 3) for the development set (model predictions).
    NeuralNet: A NeuralNet object.
    """
    train_mean = torch.mean(train_set)
    train_std = torch.std(train_set)
    train_set = (train_set - train_mean) / train_std

    in_size = 3
    out_size = 4

    net = NeuralNet(lrate=0.005, loss_fn=nn.CrossEntropyLoss(), in_size=in_size, out_size=out_size)
    train_set_reshaped = train_set.view(-1, 3, 31, 31)
    dataset = get_dataset_from_arrays(train_set_reshaped, train_labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    losses = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        for data in dataloader:
            X_batch, y_batch = data['features'], data['labels']
            batch_loss = net.step(X_batch, y_batch)
            epoch_loss += batch_loss

        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        # print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")

    dev_mean = torch.mean(dev_set)
    dev_std = torch.std(dev_set)
    dev_set = (dev_set - dev_mean) / dev_std
    dev_set_reshaped = dev_set.view(-1, 3, 31, 31)
    with torch.no_grad():
        dev_predictions = net(dev_set_reshaped)
        _, predicted_labels = torch.max(dev_predictions, 1)

    return losses, predicted_labels.numpy().astype(int), net
