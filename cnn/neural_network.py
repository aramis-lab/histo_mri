import torch
import numpy as np
import torch.nn as nn
from time import time


class HistoNet(nn.Module):

    def __init__(self):
        super(HistoNet, self).__init__()

        # CNN state

        self._currently_training = False
        self._trained = False

        # Loss of CNN
        self.criterion = nn.CrossEntropyLoss()

        # Layers
        self.conv1 = nn.Conv2d(in_channels=5, out_channels=4, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, return_indices=False,
                                     ceil_mode=False)

        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, return_indices=False,
                                     ceil_mode=False)

        self.FC1 = nn.Linear(in_features=8 * 8 * 8, out_features=128)
        self.relu3 = nn.ReLU()

        self.FC2 = nn.Linear(in_features=128, out_features=2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = x.view(-1, x[0].numel())
        x = self.FC1(x)
        x = self.relu3(x)
        x = self.FC2(x)
        x = self.softmax(x)
        return x
