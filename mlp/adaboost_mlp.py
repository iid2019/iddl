"""
Hyperparameters:
- learning rate
- number of training epoch for every single base classifier
- number of base classifiers
"""

import sys
sys.path.append('../resnet')
sys.path.append('../util')
from ensemble import AdaBoostClassifier
from models import Mlp
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import math
import time
from time_utils import format_time


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def mlpClassifier(dataloader, sample_weight_array, log_interval=200):
    begin = time.time()

    # Create the net
    input_dim = 28 * 28 * 1
    output_dim = 10
    net = Mlp(input_dim, output_dim)

    # Set device of net
    net.to(device)

    # Define the optimizer and loss function
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.5)
    criterion = nn.CrossEntropyLoss()

    n_epoch = 1
    for epoch in range(1, n_epoch + 1):
        train(net, optimizer, criterion, dataloader, sample_weight_array, epoch, log_interval=100)
        sys.stdout.write('\n')
        sys.stdout.flush()

    # Set net to evaluation mode
    net.eval()

    end = time.time()
    print('    MlpClassifier trained in {}.'.format(format_time(end - begin)))

    return net


def train(model, optimizer, criterion, dataloader, sample_weight_array, epoch, log_interval=200):
    # Set net to training mode
    model.train()

    # min_weight = sample_weight_array.min()

    # Loop over each batch from the training set
    for batch_index, (data, target) in enumerate(dataloader):
        weight = sample_weight_array[batch_index]
        # num_repeat = math.ceil(weight / min_weight)

        # Copy data to GPU if needed
        data = data.to(device)
        target = target.to(device)

        # Too slow!
        # for i in range(num_repeat):
        #     # Zero gradient buffers
        #     optimizer.zero_grad() 
                
        #     # Pass data through the network
        #     output = model(data)

        #     # Calculate loss
        #     loss = criterion(output, target)

        #     # Backpropagate
        #     loss.backward()
            
        #     # Update weights
        #     optimizer.step()
        
        # Zero gradient buffers
        optimizer.zero_grad() 
        
        # Pass data through the network
        output = model(data)

        # Calculate loss
        loss = criterion(output, target)

        # Apply the sample weight
        loss *= calculateLossFactor(weight, sample_weight_array)

        # Backpropagate
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        if (batch_index + 1) % log_interval == 0:
            sys.stdout.write('\r')
            sys.stdout.flush()
            print('    Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_index + 1) * len(data), len(dataloader.dataset),
                100. * (batch_index + 1) / len(dataloader), loss.data.item()), end='')


def calculateLossFactor(weight, weight_array):
    min = weight_array.min()
    max = weight_array.max()

    # Check if min equals max
    if min == max:
        return 1

    length = max - min
    factor = (weight - min) * 9 / length + 1
    return factor


trainset = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor())
trainloader = torch.utils.data.DataLoader(dataset=trainset, shuffle=False)

CLASSIFIER_NUM = 9
classifier = AdaBoostClassifier(mlpClassifier)
classifier.train(trainloader, classifier_num=CLASSIFIER_NUM)

test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
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

# Test the base classifier
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
    print('Test dataset: Base classifier #{} accuracy: {}/{} ({:.2f}%)'.format(i, correct, len(test_dataloader.dataset), accuracy * 100.0))
