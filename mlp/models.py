r"""Definition of the multi layer perceptron model and its variants.
"""


import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import sys
sys.path.append('../bibd')
from bibd_layer import BibdLinear, RandomSparseLinear


class Mlp(nn.Module):
    r""""Multi layer perceptron."""
    
    
    name = 'MLP'

    
    def __init__(self, input_dim, output_dim):
        super(Mlp, self).__init__()

        self.input_dim = input_dim

        # Layer definitions
        self.fc1 = nn.Linear(input_dim, 28*4)
        self.fc2 = nn.Linear(28*4, 56)
        self.fc3 = nn.Linear(56, output_dim)

    
    def forward(self, x):
        x = x.view(-1, self.input_dim)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(self.fc3(x), dim=1)
    

class BibdMlp(nn.Module):
    r"""Multi layer perceptron with BIBD."""


    name = 'B-MLP'

    
    def __init__(self, input_dim, output_dim):
        super(BibdMlp, self).__init__()

        self.input_dim = input_dim
        
        # Layer definitions
        self.bibd1 = BibdLinear(input_dim, 28*4)
        self.bibd2 = BibdLinear(28*4, 56)
        self.fc3 = nn.Linear(56, output_dim)


    def forward(self, x):
        x = x.view(-1, self.input_dim)
        x = F.relu(self.bibd1(x))
        x = F.relu(self.bibd2(x))
        return F.log_softmax(self.fc3(x), dim=1)


class RandomSparseMlp(nn.Module):
    r"""Multi layer perceptron with random sparsification."""


    name = 'R-MLP'


    def __init__(self, input_dim, output_dim):
        super(RandomSparseMlp, self).__init__()

        self.input_dim = input_dim
        
        # Layer definitions
        self.randomSparseLinear1 = RandomSparseLinear(input_dim, 28*4)
        self.randomSparseLinear2 = RandomSparseLinear(28*4, 56)
        self.fc3 = nn.Linear(56, output_dim)


    def forward(self, x):
        x = x.view(-1, self.input_dim)
        x = F.relu(self.randomSparseLinear1(x))
        x = F.relu(self.randomSparseLinear2(x))
        return F.log_softmax(self.fc3(x), dim=1)


class BaseMlp(nn.Module):
    r"""Base class for MLP based models."""


    def __init__(self, input_dim, output_dim, layers, linearLayer, name=None):
        """ Constructor.

        Args:
            input_dim: int. The dimension of input.
            output_dim: int. The dimension of output. Usually the number of categories for a classification problem.
            layers: list of int. Each number in the list represents the number of neurons in the corresponding hidden layer.
            linearLayer: nn.Linear or its subclass. For example: nn.Linear, BibdLinear, RandomSparseLinear.
            name: str. The name of this MLP.
        """

        # Set the name if necessary
        if name is not None:
            self.name = name

        super(BaseMlp, self).__init__()


        self.input_dim = input_dim
        
        # Layer definitions
        self.hiddenLinearLayers = nn.ModuleList([])
        for index, dim in enumerate(layers):
            # Get the dimension of the previous layer
            previousDim = input_dim if index == 0 else layers[index - 1]

            # Define the current hidden layer
            self.hiddenLinearLayers.append(linearLayer(previousDim, dim))
        self.outputLinearLayer = nn.Linear(layers[-1], output_dim)


    def forward(self, x):
        x = x.view(-1, self.input_dim)
        for layer in self.hiddenLinearLayers:
            x = F.relu(layer(x))
        return F.log_softmax(self.outputLinearLayer(x), dim=1)


layers3 = [28*4, 56]
layers5 = [28*4, 56, 56, 56]
layers7 = [28*4, 56, 56, 128, 128, 56]


def Mlp3(input_dim, output_dim, name='MLP-3'):
    return BaseMlp(input_dim, output_dim, layers3, nn.Linear, name=name)


def BibdMlp3(input_dim, output_dim, name='B-MLP-3'):
    return BaseMlp(input_dim, output_dim, layers3, BibdLinear, name=name)


def RandomSparseMlp3(input_dim, output_dim, name='R-MLP-3'):
    return BaseMlp(input_dim, output_dim, layers3, RandomSparseLinear, name=name)


def Mlp5(input_dim, output_dim, name='MLP-5'):
    return BaseMlp(input_dim, output_dim, layers5, nn.Linear, name=name)


def BibdMlp5(input_dim, output_dim, name='B-MLP-5'):
    return BaseMlp(input_dim, output_dim, layers5, BibdLinear, name=name)


def RandomSparseMlp5(input_dim, output_dim, name='R-MLP-5'):
    return BaseMlp(input_dim, output_dim, layers5, RandomSparseLinear, name=name)


def Mlp7(input_dim, output_dim, name='MLP-7'):
    return BaseMlp(input_dim, output_dim, layers7, nn.Linear, name=name)


def BibdMlp7(input_dim, output_dim, name='B-MLP-7'):
    return BaseMlp(input_dim, output_dim, layers7, BibdLinear, name=name)


def RandomSparseMlp7(input_dim, output_dim, name='R-MLP-7'):
    return BaseMlp(input_dim, output_dim, layers7, RandomSparseLinear, name=name)
