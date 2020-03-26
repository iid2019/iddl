
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

    min_weight = sample_weight_array.min()

    # Loop over each batch from the training set
    for batch_index, (data, target) in enumerate(dataloader):
        weight = sample_weight_array[batch_index]
        num_repeat = math.ceil(weight / min_weight)

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

        loss *= num_repeat

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


trainset = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor())
# trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=32, shuffle=True)
trainloader = torch.utils.data.DataLoader(dataset=trainset, shuffle=False)

# print(len(trainloader))

# train_data_list = []
# train_labels_list = []
# for batch_index, (inputs, targets) in enumerate(trainloader):
#     # inputs, targets = inputs.to(device), targets.to(device)
#     # print(batch_index)
#     # print(inputs.shape)
#     # print(targets.shape)

#     train_data_list.append(inputs)
#     train_labels_list.append(targets)
# train_data = torch.cat(train_data_list, dim=0)
# train_labels = torch.cat(train_labels_list, dim=0)
# print(train_data.shape)
# print(train_labels.shape)


# for index, (x, y) in enumerate(zip(train_data, train_labels)):
#     print(x, y)

# base_classifier = Mlp(input_dim, output_dim).to(device)
# classifier = AdaBoostClassifier(base_classifier)
classifier = AdaBoostClassifier(mlpClassifier)
classifier.train(trainloader, classifier_num=3)

test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
test_dataloader = torch.utils.data.DataLoader(test_dataset, shuffle=False)

val_loss, correct = 0, 0
criterion = nn.CrossEntropyLoss()
for batch_index, (data, target) in enumerate(test_dataloader):
    # Copy data to GPU if needed
    data = data.to(device)
    target = target.to(device)

    output = classifier.predict(data)
    val_loss += criterion(output, target).data.item()
    predicted = output.data.max(1)[1] # Get the index of the max log-probability
    correct += predicted.eq(target.data).cpu().sum()
test_loss /= len(test_dataloader)

accuracy = correct.to(torch.float32) / len(test_dataloader.dataset)
        
print('\nTest dataset: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_dataloader.dataset), accuracy * 100.0))
