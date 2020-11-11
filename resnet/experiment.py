"""
Copyright 2019 - 2020, Jianfeng Hou, houjf@shanghaitech.edu.cn

All rights reserved.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import time
import sys
sys.path.append('../util')
from time_utils import format_time
from term_utils import bcolors


class Experiment():
    def __init__(self, n_epoch, gpu_index=0):
        self.n_epoch = n_epoch
        self.loss_ndarray = np.zeros((0, n_epoch), dtype=float)
        self.acc_ndarray = np.zeros((0, n_epoch), dtype=float)
        self.time_array = np.array([])
        self.model_name_array = np.array([])

        self.color_list = ['#22a7f0', '#cf000f', '#03a678']

        if torch.cuda.is_available():
            self.device = torch.device("cuda:{:d}".format(gpu_index))
        else:
            self.device = torch.device('cpu')
            
        print('Using PyTorch version:', torch.__version__, ' Device:', self.device)

        return

    
    def run_model(self, model, train_loader, validation_loader, name=None, log_interval=500):
        # Store the name of the model
        if name is None:
            name = model.name
        self.model_name_array = np.append(self.model_name_array, name)

        length = len(name) + 19 + 8
        print(bcolors.OKBLUE + '<' * length + bcolors.ENDC)
        print('    ' + 'Running the model: {}'.format(name) + '    ')
        print(bcolors.OKBLUE + '-' * length + bcolors.ENDC)

        # Define the optimizer and loss function
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
        criterion = nn.CrossEntropyLoss()

        # The loss and accuracy list for this model
        loss_list, acc_list = [], []
        
        begin_time = time.time() # Begin time
        for epoch in range(1, self.n_epoch + 1):
            self.__train(model, optimizer, criterion, train_loader, epoch, log_interval=log_interval)
            self.__validate(model, criterion, validation_loader, loss_list, acc_list)
        end_time = time.time() # End time

        # Store the time
        self.time_array = np.append(self.time_array, end_time - begin_time)

        # Store the loss and accuracy data
        self.loss_ndarray.resize((self.loss_ndarray.shape[0] + 1, self.loss_ndarray.shape[1]), refcheck=False)
        self.loss_ndarray[self.loss_ndarray.shape[0] - 1] = np.array(loss_list)
        self.acc_ndarray.resize((self.acc_ndarray.shape[0] + 1, self.acc_ndarray.shape[1]), refcheck=False)
        self.acc_ndarray[self.acc_ndarray.shape[0] - 1] = np.array(acc_list)

        # Print total time usage
        time_str = format_time(end_time - begin_time)
        length = len(name) + len(time_str) + 23 + 4
        print(bcolors.OKGREEN + '-' * length + bcolors.ENDC)
        print('  ' + 'Time usage for model '+ name + ': ' + time_str + '  ')
        print(bcolors.OKGREEN + '>' * length + bcolors.ENDC)
        print()

        torch.cuda.empty_cache()
        # model = model.cpu()
        del model, train_loader, validation_loader, optimizer, criterion
        torch.cuda.empty_cache()
        # print('CUDA memory cache released.')

        # print('CUDA memory allocated: {:.2f}MB'.format(torch.cuda.memory_allocated() / 1024))
        # print('CUDA memory cached: {:.2f}MB'.format(torch.cuda.memory_cached() / 1024))

        # This is PyTorch 1.3.1, which does not support torch.cuda.memory_summary()
        # print(torch.cuda.memory_summary())

        return


    def __train(self, model, optimizer, criterion, train_loader, epoch, log_interval=200):
        # Set model to training mode
        model.train()
        
        # Loop over each batch from the training set
        for batch_idx, (data, target) in enumerate(train_loader):
            # Copy data to GPU if needed
            data = data.to(self.device)
            target = target.to(self.device)

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
            
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.data.item()))
            
            del data, target, output, loss

        del model, optimizer, criterion, train_loader


    @torch.no_grad()
    def __validate(self, model, criterion, validation_loader, loss_list, acc_list):
        model.eval()
        val_loss, correct = 0, 0
        for data, target in validation_loader:
            data = data.to(self.device)
            target = target.to(self.device)
            output = model(data)
            val_loss += criterion(output, target).data.cpu().item()
            pred = output.data.max(1)[1] # Get the index of the max log-probability
            correct += pred.eq(target.data).cpu().sum()

            del data, target, output

        val_loss /= len(validation_loader)
        loss_list.append(val_loss)

        accuracy = correct.to(torch.float32) / len(validation_loader.dataset)
        acc_list.append(accuracy)
        
        print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            val_loss, correct, len(validation_loader.dataset), accuracy * 100.0))

        del model, criterion, validation_loader


    def plot(self, loss_title='Loss v.s. Epoch', acc_title='Accuracy v.s. Epoch'):
        fig_loss, ax = plt.subplots(figsize=(10,6), dpi=200)
        plt.setp(ax.get_xticklabels(), fontsize=18, fontweight="normal")
        plt.setp(ax.get_yticklabels(), fontsize=18, fontweight="normal")
        for i in range(self.model_name_array.shape[0]):
            ax.plot(range(1, self.n_epoch + 1), self.loss_ndarray[i], '-o', color=self.color_list[i], label=self.model_name_array[i])
            ax.legend(fontsize=20)
            ax.set_title(loss_title, fontsize=24)
            ax.set_xlabel('Epoch', fontsize=20)
            ax.set_ylabel('Loss', fontsize=20)

        fig_acc, ax = plt.subplots(figsize=(10,6), dpi=200)
        plt.setp(ax.get_xticklabels(), fontsize=18, fontweight="normal")
        plt.setp(ax.get_yticklabels(), fontsize=18, fontweight="normal")
        for i in range(self.model_name_array.shape[0]):
            ax.plot(range(1, self.n_epoch + 1), self.acc_ndarray[i], '-o', color=self.color_list[i], label=self.model_name_array[i])
            ax.legend(fontsize=20)
            ax.set_title(acc_title, fontsize=24)
            ax.set_xlabel('Epoch', fontsize=20)
            ax.set_ylabel('Accuracy', fontsize=20)

        return fig_loss, fig_acc
