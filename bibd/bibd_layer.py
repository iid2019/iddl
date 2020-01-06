import torch
from torch.autograd import Variable, Function
import torch.nn as nn
import numpy as np


class BibdLinearFunction(Function):
    @staticmethod
    def forward(cxt, input, weight, mask):
        cxt.save_for_backward(input, weight, mask)
        extendWeight = weight.clone()
        extendWeight.mul_(mask.data)
        output = input.mm(extendWeight.t())
        return output


    @staticmethod
    def backward(cxt, grad_output):
        input, weight, mask = cxt.saved_tensors
        grad_input = grad_weight = grad_bias = None
        extendWeight = weight.clone()
        extendWeight.mul_(mask.data)

        if cxt.needs_input_grad[0]:
            grad_input = grad_output.mm(extendWeight)
        if cxt.needs_input_grad[1]:
            grad_weight = grad_output.clone().t().mm(input)
            grad_weight.mul_(mask.data)

        return grad_input, grad_weight, grad_bias


class BibdLinear(torch.nn.Module):
    def __init__(self, input_features, output_features, number_of_block):
        super(BibdLinear, self).__init__()
        self.input_features = input_features
        self.output_features = output_features

        self.weight = nn.Parameter(data=torch.Tensor(output_features, input_features), requires_grad=True)

        self.mask = torch.from_numpy(generate_bibd_mask(number_of_block))

        self.mask =  self.mask.cuda()
        nn.init.kaiming_normal_(self.weight.data,mode='fan_in')
        self.mask =  nn.Parameter(self.mask.cuda())
        self.mask.requires_grad = False


    def forward(self, input):
        return BibdLinearFunction.apply(input, self.weight, self.mask)
        # copy = self.weight.clone()
        # copy.mul_(self.mask.data)
        # return x.matmul(copy.t())


def generate_bibd_mask(q):
    '''
    Given q as a prime power, generate mask with size (q*(q+1), q*q)
    For example when q = 2, the mask is 
    [[1. 1. 0. 0.]
     [0. 0. 1. 1.]
     [1. 0. 1. 0.]
     [0. 1. 0. 1.]
     [1. 0. 0. 1.]
     [0. 1. 1. 0.]]
    '''

    mask = np.zeros([q * (q + 1), q * q])

    allgrids = []
    for k in range(1, q):
        grid = []
        for i in range(q):
            row = []
            for j in range(q):
                a = ((k * i + j) % q) + (q * i)
                row.append(a)
            grid.append(row)
        mols = np.array(grid).T
        allgrids.append(mols)

    for m in range(q):
        for n in range(q * m, q * m + q):
            mask[m][n] = 1

    for m in range(q, q * 2):
        for n in range(q):
            mask[m][(m - q) + q * n] = 1

    for m in range(q - 1):
        for n in range(q):
            for o in range(q):
                mask[q * (m + 2) + n][allgrids[m][n][o]] = 1
                
    return mask
