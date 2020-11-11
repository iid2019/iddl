"""
Copyright 2019 - 2020, Jianfeng Hou, houjf@shanghaitech.edu.cn

All rights reserved.

Hyperparameters:
- learning rate
- number of training epoch for every single base classifier
- number of base classifiers
- batch size
"""

from ensemble import AdaBoostClassifier
from models.sparse_resnet_v import create_resnet
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import math
import time
import sys
sys.path.append('../util')
from time_utils import format_time
import numpy as np
from art import tprint
import argparse
from os import path
import pickle


tprint('IDDL', font='larry3d')

# Parse the command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', type=str, help='The filename of the pickle file.')
parser.add_argument('-n', '--num', type=int, default=1, help='The number of base classifiers used in the enbemble.')
args = parser.parse_args()
FILENAME = args.file
NUM_CLASSIFIER = args.num

# The hyperparameters
LEARNING_RATE = 0.01
N_EPOCH = 10
BATCH_SIZE = 128
BASE_CLASSIFIER_NAME = 'B-ResNet-V-4'
print('Hyperparameters:')
print('    NUM_CLASSIFIER: {}'.format(NUM_CLASSIFIER))
print('    FILENAME: {}'.format(FILENAME))
print('    LEARNING_RATE: {}'.format(LEARNING_RATE))
print('    N_EPOCH: {}'.format(N_EPOCH))
print('    BATCH_SIZE: {}'.format(BATCH_SIZE))
print('    BASE_CLASSIFIER_NAME: {}'.format(BASE_CLASSIFIER_NAME))


begin_time = time.time()

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def BResNetVClassifier(dataloader, sample_weight_array, log_interval=200):
    begin = time.time()

    # Create the net
    net = create_resnet('18', sparsification='bibd', num_groups=4, name=BASE_CLASSIFIER_NAME)

    # Set device of net
    net.to(device)

    # Define the optimizer and loss function
    optimizer = torch.optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=0.5)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, N_EPOCH + 1):
        train(net, optimizer, criterion, dataloader, sample_weight_array, epoch, log_interval=log_interval)
        sys.stdout.write('\n')
        sys.stdout.flush()

    # Set net to evaluation mode
    net.eval()

    end = time.time()
    print('    {} trained in {}.'.format(net.name, format_time(end - begin)))

    return net


def train(model, optimizer, criterion, dataloader, sample_weight_array, epoch, log_interval=200):
    # Set net to training mode
    model.train()

    # min_weight = sample_weight_array.min()

    # Loop over each batch from the training set
    for batch_index, (data, target) in enumerate(dataloader):
        # Get the maximum weight among all the rows of this batch
        row_num = data.size()[0]
        start_index = batch_index * BATCH_SIZE
        end_index = start_index + row_num
        # # TODO
        # print(start_index, end_index)
        weight = np.amax(sample_weight_array[start_index:end_index])

        # weight = sample_weight_array[batch_index]
        num_repeat = math.floor(normalizeWeightInRange(weight, sample_weight_array, 1, 2))
        # num_repeat = math.floor(len(dataloader) * weight)

        # Copy data to GPU if needed
        data = data.to(device)
        target = target.to(device)

        if 0 == num_repeat:
            print('No need to run this sample')
            continue

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
            print('    Train epoch: {:3d} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                epoch, (batch_index + 1) * len(data), len(dataloader.dataset),
                100. * (batch_index + 1) / len(dataloader), loss.data.item()), end='')


def normalizeWeightInRange(weight, weight_array, range_left, range_right):
    """
    Normalize the weight to range [range_left, range_right]
    """
    min = weight_array.min()
    max = weight_array.max()

    # Check if min equals max
    if min == max:
        return range_left

    length = max - min
    num_repeat = (weight - min) * (range_right - range_left) / length + range_left
    return num_repeat


transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=False)
validation_dataloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False)

# Define and train the AdaBoostClassifier
classifier = AdaBoostClassifier(BResNetVClassifier, BASE_CLASSIFIER_NAME)
classifier.train(train_dataloader, validation_dataloader, classifier_num=NUM_CLASSIFIER)

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

# Test the base classifier
print('Testing all the base classifiers...')
for i in range(NUM_CLASSIFIER):
    correct = 0
    for batch_index, (data, target) in enumerate(test_dataloader):
        # Copy data to GPU if needed
        data = data.to(device)
        target = target.to(device)

        category = classifier.predict_using_base_classifier(i, data)
        target_category = target.cpu().numpy().item()
        correct += 1 if category == target_category else 0
    accuracy = correct / len(test_dataloader.dataset)
    print('Test dataset: {} #{} accuracy: {}/{} ({:.2f}%)'.format(classifier.base_classifier_name, i + 1, correct, len(test_dataloader.dataset), accuracy * 100.0))

# Test the AdaBoostClassifier
print('Testing the AdaBoostClassifier...')
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

# Persiste the result
if path.exists(FILENAME):
    result_dict = pickle.load(open(FILENAME, "rb"))
else:
    result_dict = {}
result_dict[NUM_CLASSIFIER] = accuracy
pickle.dump(result_dict, open(FILENAME, "wb"))
print('result_dict:')
for num_classifier in result_dict:
    print('    {:d}: {:.4f}'.format(num_classifier, result_dict[num_classifier]))
print('result_dict has been pickled to file \'{}\'.'.format(FILENAME))

end_time = time.time()
print('Total time usage: {}'.format(format_time(end_time - begin_time)))
