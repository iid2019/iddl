'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar
from utils import format_time
from models.r_resnet import RResNet18

import time
import numpy as np

from experiment import Experiment

import pickle


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device: {}'.format(device))


best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
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
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
net = RResNet18()
net = net.to(device)
model_name = 'R-ResNet-18' # the records will be saved in 

print(net)

# This may not work on the CS280 AI cluster
if device == 'cuda':
    print('Running using torch.nn.DataParallel...')
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True


experiment = Experiment(n_epoch=30)
experiment.run_model(net, trainloader, testloader, name='R-ResNet-18')

# Save the data record
pickle.dump(experiment.loss_ndarray, open('loss_ndarray.p', "wb"))
print('experiment.loss_ndarray dumped to file: loss_ndarray.p')
pickle.dump(experiment.acc_ndarray, open('acc_ndarray.p', "wb"))
print('experiment.loss_ndarray dumped to file: acc_ndarray.p')

fig_loss, fig_acc = experiment.plot(loss_title='Training loss v.s. Epoch on CIFAR10', acc_title='Test accuracy v.s. Epoch on CIFAR10')


# Save the plots
import matplotlib.pyplot as plt


fig_loss.set_size_inches((16, 12))
fig_loss.set_dpi(100)
fig_acc.set_size_inches((16, 12))
fig_acc.set_dpi(100)

fig_loss.savefig('fig_loss_rresnet_cifar10.eps', format='eps', pad_inches=0)
fig_acc.savefig('fig_acc_rresnet_cifar10.eps', format='eps', pad_inches=0)
