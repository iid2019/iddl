"""
Hyperparameters:
- learning rate
- number of training epoch for every single base classifier
- number of base classifiers
"""

import sys
sys.path.append('../util')
from ensemble import AdaBoostClassifier
from models.resnet_bibd_gc import ResNet_bibd_gc
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import math
import time
from time_utils import format_time

import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Parameters for adaboost')
parser.add_argument('--epoch', default=1, type=int, help='the number of the exits')
parser.add_argument('--file', default=1, type=int, help='the name of the file the data saved')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def ResNetBibdGcClassifier(dataloader, sample_weight_array, log_interval=200):
    begin = time.time()

    # Create the net
    net = ResNet_bibd_gc()

    # Set device of net
    net.to(device)

    # Define the optimizer and loss function
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.5)
    criterion = nn.CrossEntropyLoss()

    n_epoch = args.epoch
    for epoch in range(1, n_epoch + 1):
        train(net, optimizer, criterion, dataloader, sample_weight_array, epoch, log_interval=100)
        sys.stdout.write('\n')
        sys.stdout.flush()

    # Set net to evaluation mode
    net.eval()

    end = time.time()
    print('    ResNetBibdGcClassifier trained in {}.'.format(format_time(end - begin)))

    return net


def train(model, optimizer, criterion, dataloader, sample_weight_array, epoch, log_interval=200):
    # Set net to training mode
    model.train()

    # min_weight = sample_weight_array.min()

    # Loop over each batch from the training set
    for batch_index, (data, target) in enumerate(dataloader):
        weight = sample_weight_array[batch_index]
        num_repeat = math.floor(normalizeWeightInRange(weight, sample_weight_array, 10))

        # Copy data to GPU if needed
        data = data.to(device)
        target = target.to(device)

        # Repeat the sample according to the weight
        for i in range(num_repeat):
            # Zero gradient buffers
            optimizer.zero_grad() 
                
            # Pass data through the network
            output = model(data)

            # Calculate loss
            loss = criterion(output, target)

            # Backpropagate
            loss.backward()
            
            # Update weights
            optimizer.step()
        
        # # Zero gradient buffers
        # optimizer.zero_grad() 
        
        # # Pass data through the network
        # output = model(data)

        # # Calculate loss
        # loss = criterion(output, target)

        # # Apply the sample weight
        # loss *= calculateLossFactor(weight, sample_weight_array)

        # # Backpropagate
        # loss.backward()
        
        # # Update weights
        # optimizer.step()
        
        if (batch_index + 1) % log_interval == 0:
            sys.stdout.write('\r')
            sys.stdout.flush()
            print('    Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_index + 1) * len(data), len(dataloader.dataset),
                100. * (batch_index + 1) / len(dataloader), loss.data.item()), end='')


def normalizeWeightInRange(weight, weight_array, range_right):
    """
    Normalize the weight to range [1, range_right]
    """
    min = weight_array.min()
    max = weight_array.max()

    # Check if min equals max
    if min == max:
        return 1

    length = max - min
    num_repeat = (weight - min) * (range_right - 1) / length + 1
    return num_repeat


transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, shuffle=False)

CLASSIFIER_NUM = 5
classifier = AdaBoostClassifier(ResNetBibdGcClassifier)
classifier.train(trainloader, classifier_num=CLASSIFIER_NUM)

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
test_dataloader = torch.utils.data.DataLoader(test_dataset, shuffle=False)

# Test the AdaBoostClassifier
correct = 0
for batch_index, (data, target) in enumerate(test_dataloader):
    # Copy data to GPU if needed
    data = data.to(device)
    target = target.to(device)

    category = classifier.predict(data)
    target_category = target.cpu().numpy().item()
    correct += 1 if category == target_category else 0
accuracy = correct / len(test_dataloader.dataset)
print('\nTest dataset: AdaBoostClassifier accuracy: {}/{} ({:.2f}%)\n'.format(correct, len(test_dataloader.dataset), accuracy * 100.0))

test_acc = []
# Test the base classifier

CLASSIFIER_NUM = 5

for i in range(CLASSIFIER_NUM):
    correct = 0
    for batch_index, (data, target) in enumerate(test_dataloader):
        # Copy data to GPU if needed
        data = data.to(device)
        target = target.to(device)

        category = classifier.predict_using_base_classifier(i, data)
        target_category = target.cpu().numpy().item()
        correct += 1 if category == target_category else 0
    accuracy = correct / len(test_dataloader.dataset)
    test_acc.append(accuracy)
    print('Test dataset: Base ResNetBibdGcClassifier #{} accuracy: {}/{} ({:.2f}%)'.format(i + 1, correct, len(test_dataloader.dataset), accuracy * 100.0))

test_acc = np.array(test_acc)
#test_time = np.array(test_time)
np.savetxt("./results/Adaboost/test_acc"+str(args.file)+".csv", test_acc, fmt = '%.3e', delimiter = ",")