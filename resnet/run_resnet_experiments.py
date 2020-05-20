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

import time
import numpy as np

from experiment import Experiment
import pickle
from datetime import datetime
import sys
from art import tprint
from models.sparse_resnet_v import create_resnet

tprint('IDDL', font='larry3d')

# Use start time for the filename of the pickled file
# date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
print('ResNet experiments started at {}.'.format(datetime.now().strftime("%Y%m%d_%H%M%S")))


# Hyperparameters
BATCH_SIZE = 128
N_EPOCH = 50
print('Hyperparameters:')
print('    BATCH_SIZE: {:d}'.format(BATCH_SIZE))
print('    N_EPOCH: {:d}'.format(N_EPOCH))


# Dictionary for model_name to creation parameters
# Keys: str. The name of the model
# Values: tuple. (arch, sparsification, name)
model_param_dict = {
    'ResNet-18': ('18', 'none', 'ResNet-18'), 'ResNet-34': ('34', 'none', 'ResNet-34'), 'ResNet-50': ('50', 'none', 'ResNet-50'),
    'B-ResNet-18': ('18', 'bibd', 'B-ResNet-18'), 'B-ResNet-34': ('34', 'bibd', 'B-ResNet-34'), 'B-ResNet-50': ('50', 'bibd', 'B-ResNet-50'),
    'R-ResNet-18': ('18', 'random', 'R-ResNet-18'), 'R-ResNet-34': ('34', 'random', 'R-ResNet-34'), 'R-ResNet-50': ('50', 'random', 'R-ResNet-50'),
}


# For parsing command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, help='The only one model you want to run.')
parser.add_argument('-n', '--name', type=str, help='The name of the pickled files, which will be model_name_array_{name}.pkl, accuracy_array_{name}.pkl')
parser.add_argument('-g', '--gpu', type=int, default=0, help='The GPU index the program will run on. Starting from 0.')
args = parser.parse_args()
model_name = args.model
pickle_name = args.name
gpu_index = args.gpu


if torch.cuda.is_available():
    device = torch.device("cuda:{}".format(gpu_index))
    print('CUDA available. PyTorch version:', torch.__version__, ' Device:', device)
else:
    print('CUDA is not available. Stopped.')
    sys.exit()


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

experiment = Experiment(n_epoch=N_EPOCH, gpu_index=gpu_index)
params = model_param_dict[model_name]
model = create_resnet(arch=params[0], sparsification=params[1], name=params[2]).to(device)
experiment.run_model(model, trainloader, testloader)

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
