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
        self.fc1 = nn.Linear(input_dim, 49)
        self.fc1_drop = nn.Dropout(0.2)
        self.fc2 = nn.Linear(49, 56)
        self.fc3 = nn.Linear(56, output_dim)

    
    def forward(self, x):
        x = x.view(-1, self.input_dim)
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = F.relu(self.fc2(x))
        return F.log_softmax(self.fc3(x), dim=1)
    

class BibdMlp(nn.Module):
    r"""Multi layer perceptron with BIBD."""


    name = 'MLP with BIBD'

    
    def __init__(self, input_dim, output_dim):
        super(BibdMlp, self).__init__()

        self.input_dim = input_dim
        
        # Layer definitions
        q = 7 # Number of blocks
        self.fc1 = nn.Linear(input_dim, q*q)
        self.fc1_drop = nn.Dropout(0.2)
        self.bibd2 = BibdLinear(q*q, q*(q+1))
        self.fc3 = nn.Linear(q*(q+1), output_dim)


    def forward(self, x):
        x = x.view(-1, self.input_dim)
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = F.relu(self.bibd2(x))
        return F.log_softmax(self.fc3(x), dim=1)


class RandomSparseMlp(nn.Module):
    r"""Multi layer perceptron with random sparsification."""


    name = 'Random sparse MLP'


    def __init__(self, input_dim, output_dim):
        super(RandomSparseMlp, self).__init__()

        self.input_dim = input_dim
        
        # Layer definitions
        q = 7
        self.fc1 = nn.Linear(input_dim, q*q)
        self.fc1_drop = nn.Dropout(0.2)
        self.randomSparseLinear2 = RandomSparseLinear(q*q, q*(q+1))
        self.fc3 = nn.Linear(q*(q+1), output_dim)


    def forward(self, x):
        x = x.view(-1, self.input_dim)
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = F.relu(self.randomSparseLinear2(x))
        return F.log_softmax(self.fc3(x), dim=1)
