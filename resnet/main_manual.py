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
import matplotlib.pyplot as plt

from models import *
from utils import progress_bar
from utils import format_time
from models.resnet_bibd import *
from models.resnet_gc import *
from models.resnet_bibd_gc import *
from models.resnet_exit import *
from models.resnet_exit_BIBD import *
from models.resnet_exit_gc import *
from models.resnet_ENS_EE import *
from models.resnet_ENS_BIBD_EE import *
from models.resnet_BIBD_EE_GC import *

import time
import numpy as np

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--en', default=3, type=int, help='the number of the exits')
parser.add_argument('--epoch', default=30, type=int, help='the number of the exits')
parser.add_argument('--file', default=1, type=int, help='the name of the file the data saved')
args = parser.parse_args()
num_exit = args.en

device = 'cpu'
dtype = torch.float32

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
net = ResNet_BIBD_EE_GC()
# net = ResNet_e_B() # ResNet with the early exit and BIBD
net = net.to(device)

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
    net.train()
    correct = np.zeros(num_exit)
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        
        loss = criterion(outputs[2], targets)
        # loss.backward(retain_graph = True)
        loss.backward(retain_graph = True)
        optimizer.step()
        
        outputs[0] = e1_Net(outputs[0], params1) # the outputs of the exits branches
        outputs[1] = e2_Net(outputs[1], params2)
        
        # add comment to check the main exit accuracy
        
        loss0 = F.cross_entropy(outputs[0], targets)
        loss0.backward(retain_graph = True)
        
        loss1 = F.cross_entropy(outputs[1], targets)
        loss1.backward()
        
        with torch.no_grad():
            for w in params1 + params2:
                w -= args.lr * w.grad

                # Manually zero the gradients after running the backward pass
                w.grad.zero_()

        for i in range(num_exit):
                _, predicted = outputs[i].max(1)
                correct[i] += predicted.eq(targets).sum().item()
        
        total += targets.size(0)
        
        msg = ''
        for i in range(num_exit):
            msg = msg + '| Ex%d: %.2f%%' % (i + 1, 100. * correct[i] / total)

        progress_bar(batch_idx, len(trainloader), msg)
        
        # record the result
    records += [[loss.data.tolist(), correct[2]/total, loss0.data.tolist(), correct[0]/total, loss1.data.tolist(), correct[1]/total]]
        

def test(epoch, records, test_time):
    global best_acc
    net.eval()
    test_loss = 0
    correct = np.zeros(num_exit)
    total = 0
    #time_ = time.time()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            #time_0 = time.time()
            outputs[0] = e1_Net(outputs[0], params1)
            #time_1 = time.time()
            outputs[1] = e2_Net(outputs[1], params2)
            #time_2 = time.time()
            
            loss = criterion(outputs[2], targets)
            loss0 = criterion(outputs[0], targets)
            loss1 = criterion(outputs[1], targets)
            
            for i in range(num_exit):
                _, predicted = outputs[i].max(1)
                correct[i] += predicted.eq(targets).sum().item()
                
            total += targets.size(0)
         
        msg = ''
        for i in range(num_exit):
            msg = msg + '| Ex%d Acc: %.2f%%' % (i + 1, 100. * correct[i] / total)
        print(msg)
    
    # Save checkpoint
    acc = 100.0 * correct[num_exit-1] / total
    if acc > best_acc:
        print('Saving...')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc
        
    # record the result
    records += [[loss.data.tolist(), correct[2]/total, loss0.data.tolist(), correct[0]/total, loss1.data.tolist(), correct[1]/total]]
    #test_time += [[(time_0 - time_) * 1000, (time_1 - time_) * 1000, (time_2 - time_) * 1000]]
        
def flatten(x):
    N = x.shape[0] # read in N, C, H, W
    return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image

def random_weight(shape): 
    """
    Initialization
    The BIBD part can be added in this function.
    Create random Tensors for weights; setting requires_grad=True means that we
    want to compute gradients for these Tensors during the backward pass.
    We use Kaiming normalization: sqrt(2 / fan_in)
    """
    if len(shape) == 2:  # FC weight
        fan_in = shape[0]
    else:
        fan_in = np.prod(shape[1:]) # conv weight [out_channel, in_channel, kH, kW]
    # randn is standard normal distribution generator. 
    
    w = torch.randn(shape, device=device, dtype=dtype) * np.sqrt(2. / fan_in)
    w.requires_grad = True
    
    return w

def zero_weight(shape):
    return torch.zeros(shape, device=device, dtype=dtype, requires_grad=True)

def e1_Net(exit1, params):
    conv1_w, conv1_b, fc1_w, fc1_b = params
    exit1 = F.relu(F.conv2d(exit1, conv1_w, conv1_b, stride = 1, padding = 1))
    exit1 = flatten(exit1).mm(fc1_w) + fc1_b
    return exit1
    
def e2_Net(exit2, params):
    conv2_w, conv2_b, fc2_w, fc2_b = params
    exit2 = F.relu(F.conv2d(exit2, conv2_w, conv2_b, stride = 1, padding = 1))
    exit2 = flatten(exit2).mm(fc2_w) + fc2_b
    return exit2
    
# initialization
conv1_w = random_weight((32, 128, 3, 3))
conv1_b = zero_weight((32,))
fc1_w = random_weight((32*16*16, 10))
fc1_b = zero_weight((10,))
params1 = [conv1_w, conv1_b, fc1_w, fc1_b]
    
conv2_w = random_weight((32, 256, 3, 3))
conv2_b = zero_weight((32,))
fc2_w = random_weight((32*8*8, 10))
fc2_b = zero_weight((10,))
params2 = [conv2_w, conv2_b, fc2_w, fc2_b]

train_records = [] # train_records[i]: the training loss and accuracy of ith exit
test_records = []
test_time = [] # the inference time

begin_time = time.time()
for ep in range(start_epoch, start_epoch+args.epoch):
    # record the results of all three exits
    train(ep, train_records)
    test(ep, test_records, test_time)
    
end_time = time.time()
print('Total time usage: {}'.format(format_time(end_time - begin_time)))

train_records = np.array(train_records)
test_records = np.array(test_records)
#test_time = np.array(test_time)
np.savetxt("./results/BIBD_EE_GC/train_BIBD_EE_GC_"+str(args.file)+".csv", train_records, fmt = '%.3e', delimiter = ",")
np.savetxt("./results/BIBD_EE_GC/test_BIBD_EE_GC_"+str(args.file)+".csv", test_records, fmt = '%.3e', delimiter = ",")
#np.savetxt("./results/time_e_B_"+str(args.file)+".csv", test_time, fmt = '%.4e', delimiter = ",")

