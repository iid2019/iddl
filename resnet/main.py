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
from models.resnet_bibd import *
from models.resnet_gc import *
from models.resnet_bibd_gc import *
from models.resnet_exit import *
from models.resnet_ENS_BIBD import *
from models.resnet_bibd_gc import *
from models.resnet_GC_ENS_BIBD import *

import time
import numpy as np


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--en', default=1, type=int, help='the number of the exits')
parser.add_argument('--epoch', default=30, type=int, help='the number of the exits')
parser.add_argument('--file', default=1, type=int, help='the name of the file the data saved')
args = parser.parse_args()
num_exit = args.en

device = 'cpu'

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
# net = VGG('VGG19')
net = ResNet18()
# net = GEBResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = ResNeXt29_2x64d_bibd()
# net = ResNet_gc()
# net = BResNet18()
# net = ResNet_bibd_gc()
net = net.to(device)
model_name = 'ResNet18' # the records will be saved in 

print(net)

# This may not work on the CS280 AI cluster
if device == 'cuda':
    print('Running using torch.nn.DataParallel...')
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True


if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)


def train(epoch, records):
    print('\nEpoch: %d' % epoch)
    st_time = time.time()
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    en_time = time.time()
    records += [[loss.data.tolist(), correct/total, (en_time - st_time) * 1000]]

def test(epoch, records):
    global best_acc
    st_time = time.time()
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            outputs = outputs
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    en_time = time.time()
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc
    records += [[loss.data.tolist(), correct/total, (en_time - st_time) * 1000]]

train_records = [] # train_records[i]: the training loss and accuracy of ith exit
test_records = []
    
begin_time = time.time()
for ep in range(start_epoch, start_epoch+args.epoch):
    # record the results of all three exits
    train(ep, train_records)
    test(ep, test_records)
    
end_time = time.time()
print('Total time usage: {}'.format(format_time(end_time - begin_time)))

train_records = np.array(train_records)
test_records = np.array(test_records)
train_loss = train_records[:, 0]
train_acc = train_records[:, 1]
train_time = train_records[:, 2]
test_loss = test_records[:, 0]
test_acc = test_records[:, 1]
test_time = test_records[:, 2]
np.savetxt("./results/"+model_name+"/train_loss_"+str(args.file)+".csv", train_loss, fmt = '%.3e', delimiter = ",")
np.savetxt("./results/"+model_name+"/train_acc_"+str(args.file)+".csv", train_acc, fmt = '%.3e', delimiter = ",")
np.savetxt("./results/"+model_name+"/train_time_"+str(args.file)+".csv", train_time, fmt = '%.3e', delimiter = ",")
np.savetxt("./results/"+model_name+"/test_loss_"+str(args.file)+".csv", test_loss, fmt = '%.3e', delimiter = ",")
np.savetxt("./results/"+model_name+"/test_acc_"+str(args.file)+".csv", test_acc, fmt = '%.3e', delimiter = ",")
np.savetxt("./results/"+model_name+"/test_time_"+str(args.file)+".csv", test_time, fmt = '%.3e', delimiter = ",")
