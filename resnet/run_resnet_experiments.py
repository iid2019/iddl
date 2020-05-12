r'''Run BIBD experiments of ResNet based models.'''


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

import os
from os import path
import argparse

from utils import progress_bar
from utils import format_time
from models.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from models.r_resnet import RResNet18, RResNet34, RResNet50, RResNet101, RResNet152
from models.resnet_bibd import BResNet18, BResNet34, BResNet50, BResNet101, BResNet152

import time
import numpy as np

from experiment import Experiment
import pickle
from datetime import datetime
import sys


# Use start time for the filename of the pickled file
# date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
print('ResNet experiments started at {}.'.format(datetime.now().strftime("%Y%m%d_%H%M%S")))


# Hyperparameters
BATCH_SIZE = 128
N_EPOCH = 1
print('Hyperparameters:')
print('    BATCH_SIZE: {:d}'.format(BATCH_SIZE))
print('    N_EPOCH: {:d}'.format(N_EPOCH))


# Dictionary for argument to model
model_dict = {
    'ResNet-18': ResNet18, 'ResNet-34': ResNet34, 'ResNet-50': ResNet50,
    'B-ResNet-18': BResNet18, 'B-ResNet-34': BResNet34, 'B-ResNet-50': BResNet50,
    'R-ResNet-18': RResNet18, 'R-ResNet-34': RResNet34, 'R-ResNet-50': RResNet50,
}


# For parsing command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, help='The only one model you want to run.')
parser.add_argument('-n', '--name', type=str, help='The name of the pickled files, which will be model_name_array_{name}.pkl accuracy_array_{name}.pkl')
args = parser.parse_args()
model_name = args.model
pickle_name = args.name


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using PyTorch version:', torch.__version__, ' Device:', device)


print('Preparing the datasets...')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

experiment = Experiment(n_epoch=N_EPOCH)
if torch.cuda.is_available():
    experiment.run_model(model_dict[model_name]().to(device), trainloader, testloader)

    # # ResNet models
    # experiment.run_model(ResNet18().to(device), trainloader, testloader)
    # experiment.run_model(ResNet34().to(device), trainloader, testloader)
    # experiment.run_model(ResNet50().to(device), trainloader, testloader)

    # # B-ResNet models
    # experiment.run_model(BResNet18().to(device), trainloader, testloader)
    # experiment.run_model(BResNet34().to(device), trainloader, testloader)
    # experiment.run_model(BResNet50().to(device), trainloader, testloader)

    # # R-ResNet models
    # experiment.run_model(RResNet18().to(device), trainloader, testloader)
    # experiment.run_model(RResNet34().to(device), trainloader, testloader)
    # experiment.run_model(RResNet50().to(device), trainloader, testloader)
else:
    print('CUDA is not available. Stopped.')
    sys.exit()


# Save all the experiment data
# filename = 'resnet_experiments_{}.pkl'.format(date_time)
# pickle.dump(experiment, open(filename, "wb"))
# print('The Experiment instance experiment dumped to the file: {}'.format(filename))

# Persist the test accuracy
model_filename = 'model_name_array_{}.pkl'.format(pickle_name)
acc_filename = 'accuracy_array_{}.pkl'.format(pickle_name)
if path.exists(model_filename):
    model_name_array = pickle.load(open(model_filename, "rb"))
else:
    model_name_array = np.array([], dtype=str)
if path.exists(acc_filename):
    accuracy_array = pickle.load(open(acc_filename, "rb"))
else:
    accuracy_array = np.array([], dtype=float)
model_name_array = np.append(model_name_array, experiment.model_name_array[-1])
accuracy_array = np.append(accuracy_array, experiment.acc_ndarray[-1][-1])
pickle.dump(model_name_array, open(model_filename, "wb"))
pickle.dump(accuracy_array, open(acc_filename, "wb"))
print('Data in {}:'.format(model_filename))
print(model_name_array)
print('Data in {}:'.format(acc_filename))
print(accuracy_array)
print('Data has been dumped to files: {}, {}'.format(model_filename, acc_filename))

print('ResNet experiment for the model {} completed at {}.'.format(model_name, datetime.now().strftime("%Y%m%d_%H%M%S")))
