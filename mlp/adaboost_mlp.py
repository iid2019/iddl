
import sys
sys.path.append('../resnet')
from ensemble import AdaBoostClassifier
from models import Mlp
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def mlpClassifier(dataloader, sample_weight_array, log_interval=200):
    # Create the net
    input_dim = 28 * 28 * 1
    output_dim = 10
    net = Mlp(input_dim, output_dim)

    # Set device of net
    net.to(device)

    # Define the optimizer and loss function
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.5)
    criterion = nn.CrossEntropyLoss()

    n_epoch = 10
    for epoch in range(1, n_epoch + 1):
        train(net, optimizer, criterion, dataloader, epoch, log_interval=500)

    # Set net to evaluation mode
    net.eval()

    return net


def train(model, optimizer, criterion, dataloader, epoch, log_interval=200):
    # Set net to training mode
    model.train()

    # Loop over each batch from the training set
    for batch_index, (data, target) in enumerate(dataloader):
        # Copy data to GPU if needed
        data = data.to(device)
        target = target.to(device)

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
        
        if batch_index % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_index * len(data), len(dataloader.dataset),
                100. * batch_index / len(dataloader), loss.data.item()))


trainset = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor())
# trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=32, shuffle=True)
trainloader = torch.utils.data.DataLoader(dataset=trainset, shuffle=False)

# print(len(trainloader))

train_data_list = []
train_labels_list = []
for batch_index, (inputs, targets) in enumerate(trainloader):
    # inputs, targets = inputs.to(device), targets.to(device)
    # print(batch_index)
    # print(inputs.shape)
    # print(targets.shape)

    train_data_list.append(inputs)
    train_labels_list.append(targets)
train_data = torch.cat(train_data_list, dim=0)
train_labels = torch.cat(train_labels_list, dim=0)
print(train_data.shape)
print(train_labels.shape)


# for index, (x, y) in enumerate(zip(train_data, train_labels)):
#     print(x, y)

# base_classifier = Mlp(input_dim, output_dim).to(device)
# classifier = AdaBoostClassifier(base_classifier)
classifier = AdaBoostClassifier(mlpClassifier)
classifier.train(trainloader)
