import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F  # useful stateless functions
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import sampler

import torchvision.datasets as dset
import torchvision.transforms as T

from utils import progress_bar
from utils import format_time

import numpy as np

NUM_TRAIN = 49000

transform = T.Compose([
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])

# We set up a Dataset object for each split (train / val / test); Datasets load
# training examples one at a time, so we wrap each Dataset in a DataLoader which
# iterates through the Dataset and forms minibatches. We divide the CIFAR-10
# training set into train and val sets by passing a Sampler object to the
# DataLoader telling how it should sample from the underlying Dataset.
cifar10_train = dset.CIFAR10('./data', train=True, download=True,
                             transform=transform)
loader_train = DataLoader(cifar10_train, batch_size=100, 
                          sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))

cifar10_val = dset.CIFAR10('./data', train=True, download=True,
                           transform=transform)
loader_val = DataLoader(cifar10_val, batch_size=100, 
                        sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, 50000)))

cifar10_test = dset.CIFAR10('./data', train=False, download=True, 
                            transform=transform)
loader_test = DataLoader(cifar10_test, batch_size=100)

USE_GPU = True

dtype = torch.float32 # we will be using float throughout this tutorial

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Constant to control how frequently we print train loss
# print_every = 100

print('using device:', device)

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

def check_accuracy_part(loader, model_fn, params, num_exit):
    """
    Check the accuracy of a classification model.
    
    Inputs:
    - loader: A DataLoader for the data split we want to check
    - model_fn: A function that performs the forward pass of the model,
      with the signature scores = model_fn(x, params)
    - params: List of PyTorch Tensors giving parameters of the model
    - num_exit: The number of exits in the net.
    
    Returns: Nothing, but prints the accuracy of the model
    """
    num_exit += 1
    split = 'val' if loader.dataset.train else 'test'
    print('Checking accuracy on the %s set' % split)
    num_samples = 0
    num_correct = np.zeros(num_exit)
    
    with torch.no_grad(): # this is testing part step
        for x, y in loader:
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.int64)
            outputs = model_fn(x, params) # if there are many exits, the scores will be a vector
            
            for i in range(num_exit):
                _, preds = outputs[i].max(1)
                num_correct[i] += (preds == y).sum()
            num_samples += preds.size(0)
        
        acc = num_correct / float(num_samples)
        
        msg = ''
        for i in range(num_exit):
            msg = msg + '| Ex%d Acc: %.2f%%' % (i, 100. * num_correct[i] / num_samples)
        
        print(msg)
        
def train_part(model_fn, params, learning_rate, num_exit):
    """
    Train a model on CIFAR-10.
    
    Inputs:
    - model_fn: A Python function that performs the forward pass of the model.
      It should have the signature scores = model_fn(x, params) where x is a
      PyTorch Tensor of image data, params is a list of PyTorch Tensors giving
      model weights, and scores is a PyTorch Tensor of shape (N, C) giving
      scores for the elements in x.
    - params: List of PyTorch Tensors giving weights for the model
    - learning_rate: Python scalar giving the learning rate to use for SGD
    - num_exit: The number of exits in the net. 
    
    Returns: Nothing
    """
    num_exit += 1
    train_loss = 0
    num_samples = 0
    num_correct = np.zeros(num_exit)
    
    for t, (x, y) in enumerate(loader_train):
        # Move the data to the proper device (GPU or CPU)
        x = x.to(device=device, dtype=dtype)
        y = y.to(device=device, dtype=torch.long)

        # Forward pass: compute scores and loss
        outputs = model_fn(x, params) # if there are many exits, the scores will be a vector
        
        # In this case, we determine the scores by 0.9*out0 + 0.09*out1 + 0.009*out2 + ...
        # mask = np.array([9 * 10 ** (-i - 1) for i in range(num_exit)])
        # mask = np.ones(num_exit)
        scores = outputs.sum(axis = 0)

        loss = F.cross_entropy(scores, y) # this scores should be a weighted? or only the last exit?

        # Backward pass: PyTorch figures out which Tensors in the computational
        # graph has requires_grad=True and uses backpropagation to compute the
        # gradient of the loss with respect to these Tensors, and stores the
        # gradients in the .grad attribute of each Tensor.
        loss.backward()

        # Update parameters. We don't want to backpropagate through the
        # parameter updates, so we scope the updates under a torch.no_grad()
        # context manager to prevent a computational graph from being built.
        with torch.no_grad():
            for w in params:
                #print (w.shape)
                w -= learning_rate * w.grad

                # Manually zero the gradients after running the backward pass
                w.grad.zero_()

        train_loss += loss.item()
        
        for i in range(num_exit):
                _, preds = outputs[i].max(1)
                num_correct[i] += (preds == y).sum()
        num_samples += preds.size(0)
        acc = num_correct / float(num_samples)
        
        msg = 'Loss: %.2f' % (train_loss / (t + 1))
        
        for i in range(num_exit):
            msg = msg + '| Ex%d Acc: %.2f%%' % (i, 100. * num_correct[i] / num_samples)
        
        progress_bar(t, len(loader_train), msg)
        
def NaiveNet(x, params):
    """
    Performs the forward pass of a exit convolutional network with the
    architecture defined above.

    Inputs:
    - x: A PyTorch Tensor of shape (N, 3, H, W) giving a minibatch of images
    - params: A list of PyTorch Tensors giving the weights and biases for the
      network; should contain the following:
      - conv_w1: PyTorch Tensor of shape (channel_1, 3, KH1, KW1) giving weights
        for the first convolutional layer
      - conv_b1: PyTorch Tensor of shape (channel_1,) giving biases for the first
        convolutional layer
      - conv_w2: PyTorch Tensor of shape (channel_2, channel_1, KH2, KW2) giving
        weights for the second convolutional layer
      - conv_b2: PyTorch Tensor of shape (channel_2,) giving biases for the second
        convolutional layer
      - fc_w: PyTorch Tensor giving weights for the fully-connected layer. Can you
        figure out what the shape should be?
      - fc_b: PyTorch Tensor giving biases for the fully-connected layer. Can you
        figure out what the shape should be?
    
    Returns:
    - scores: PyTorch Tensor of shape (N, C) giving classification scores for x
    """
    conv_w1, conv_b1, conv_w2, conv_b2, fc_w, fc_b, e1_w, e1_b = params # explict call
    scores = None
    # print ("x.shape", x.shape) [32, 3, 32, 32]
    l1_output = F.relu ( F.conv2d(x, conv_w1, conv_b1, padding = 2) )
    # print ("l1_output.shape", l1_output.shape) [32, 32, 32, 32]
    l2_output = F.relu ( F.conv2d(l1_output, conv_w2, conv_b2, padding = 1) )
    # print ("l2_output.shape", l2_output.shape) [32, 16, 32, 32]
    scores = flatten(l2_output).mm(fc_w) + fc_b
    # print ("scores.shape", scores.shape) [32, 10]
    
    
    exit = F.relu ( F.conv2d(l1_output, conv_w2, conv_b2, padding = 1) )
    exit = flatten(exit).mm(e1_w) + e1_b
    
    return np.array([scores, exit])

learning_rate = 3e-3

channel_1 = 32
channel_2 = 16

conv_w1 = None
conv_b1 = None
conv_w2 = None
conv_b2 = None
fc_w = None
fc_b = None

# define for exit
e1_w = random_weight((16 * 32 * 32, 10))
e1_b = zero_weight((10,))


# Initialize the parameters
conv_w1 = random_weight((channel_1, 3, 5, 5))
conv_b1 = zero_weight((channel_1,))
conv_w2 = random_weight((channel_2, channel_1, 3, 3))
conv_b2 = zero_weight((channel_2,))
fc_w = random_weight((32 * 32 * 16, 10))
fc_b = zero_weight((10,))

params = [conv_w1, conv_b1, conv_w2, conv_b2, fc_w, fc_b, e1_w, e1_b]

for epoch in range(30):
    print ("Epoch %d :" %(epoch))
    train_part(NaiveNet, params, learning_rate, 1)
    check_accuracy_part(loader_val, NaiveNet, params, 1)