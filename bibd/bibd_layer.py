import torch
from torch.autograd import Variable, Function
import torch.nn as nn
import numpy as np
import math


def generate_bibd_mask(q):
    r'''
    Given q as a prime power, generates mask with size (q*(q+1), q*q)
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


def generate_fake_bibd_mask(v, b):
    r"""A fake BIBD generator via BIBD truncation."""


    # Calculate the minimal q to cover v
    q = math.ceil(math.sqrt(v))

    bibd_mask = generate_bibd_mask(q)

    mask = np.zeros([b, v])
    for row in range(mask.shape[0]):
        for col in range(mask.shape[1]):
            mask[row][col] = bibd_mask[row % (q*(q+1))][col]
                
    return mask


def generate_random_sparse_mask(v, b):
    mask = generate_fake_bibd_mask(v, b).reshape((v*b, 1))
    np.random.shuffle(mask)
    mask = mask.reshape((b, v))
    return mask


class BibdLinearFunction(Function):
    # TODO: This should be deprecated with replacement of PrunedLinearFunction
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


class PrunedLinearFunction(Function):
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
    def __init__(self, input_features, output_features):
        super(BibdLinear, self).__init__()

        self.input_features = input_features
        self.output_features = output_features

        self.weight = nn.Parameter(data=torch.Tensor(output_features, input_features), requires_grad=True)
        nn.init.kaiming_normal_(self.weight.data, mode='fan_in')

        self.mask = torch.from_numpy(generate_fake_bibd_mask(input_features, output_features))

        self.mask =  self.mask.cuda()
        self.mask =  nn.Parameter(self.mask.cuda())
        self.mask.requires_grad = False


    def forward(self, input):
        return BibdLinearFunction.apply(input, self.weight, self.mask)


class RandomSparseLinear(torch.nn.Module):
    def __init__(self, input_features, output_features):
        super(RandomSparseLinear, self).__init__()

        self.input_features = input_features
        self.output_features = output_features

        self.weight = nn.Parameter(data=torch.Tensor(output_features, input_features), requires_grad=True)
        nn.init.kaiming_normal_(self.weight.data, mode='fan_in')

        self.mask = torch.from_numpy(generate_random_sparse_mask(input_features, output_features))

        self.mask =  self.mask.cuda()
        self.mask =  nn.Parameter(self.mask.cuda())
        self.mask.requires_grad = False


    def forward(self, input):
        return PrunedLinearFunction.apply(input, self.weight, self.mask)


class MulExpander(Function):
    @staticmethod
    def forward(cxt, weight, mask):
        cxt.save_for_backward(mask)

        extendWeights = weight.clone()
        extendWeights.mul_(mask.data)

        return extendWeights

    
    @staticmethod
    def backward(cxt, grad_output):
        grad_bias = None

        mask = cxt.saved_tensors[0]

        grad_weight = grad_output.clone()
        grad_weight.mul_(mask.data)

        return grad_weight, grad_bias


class execute2DConvolution(torch.nn.Module):
    def __init__(self, mask, inStride=1, inPadding=0, inDilation=1, inGroups=1):
        super(execute2DConvolution, self).__init__()
        self.cStride = inStride
        self.cPad = inPadding
        self.cDil = inDilation
        self.cGrp = inGroups
        self.mask = mask


    def forward(self, dataIn, weightIn):
        fpWeights = MulExpander.apply(weightIn, self.mask)
        return torch.nn.functional.conv2d(dataIn, fpWeights, bias=None,
                                          stride=self.cStride, padding=self.cPad,
                                          dilation=self.cDil, groups=self.cGrp)


class BibdConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, inDil=1, groups=1):
        super(BibdConv2d, self).__init__()
        # Initialize all parameters that the convolution function needs to know
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conStride = stride
        self.conPad = padding
        self.outPad = 0
        self.conDil = inDil
        self.conTrans = False
        self.conGroups = groups

        n = kernel_size * kernel_size * out_channels
        # initialize the weights and the bias as well as the
        self.fpWeight = torch.nn.Parameter(data=torch.Tensor(out_channels, in_channels//self.conGroups, kernel_size, kernel_size), requires_grad=True)
        nn.init.kaiming_normal_(self.fpWeight.data, mode='fan_out')

        fake_bibd_mask = generate_fake_bibd_mask(in_channels, out_channels)
        self.mask = torch.zeros(out_channels, (in_channels//self.conGroups), 1, 1)
        for i in range(out_channels):
            for j in range(in_channels//self.conGroups):
                self.mask[i][j][0][0] = fake_bibd_mask[i][j]

        self.mask = self.mask.repeat(1, 1, kernel_size, kernel_size)
        self.mask = nn.Parameter(self.mask.cuda())
        self.mask.requires_grad = False


    def forward(self, dataInput):
        return execute2DConvolution(self.mask, self.conStride, self.conPad,self.conDil, self.conGroups)(dataInput, self.fpWeight)
