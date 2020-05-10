# -*- coding: UTF-8 -*-

"""
R-ResNet: Random sparse ResNet.

References:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('../bibd')
from bibd_layer import RandomSparseConv2d


class RBasicBlock(nn.Module):
    expansion = 1


    def __init__(self, in_planes, planes, stride=1):
        super(RBasicBlock, self).__init__()
        self.conv1 = RandomSparseConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = RandomSparseConv2d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )


    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class RBottleneck(nn.Module):
    expansion = 4


    def __init__(self, in_planes, planes, stride=1):
        super(RBottleneck, self).__init__()
        self.conv1 = RandomSparseConv2d(in_planes, planes, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = RandomSparseConv2d(planes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = RandomSparseConv2d(planes, self.expansion*planes, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )


    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class RResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, name='R-ResNet'):
        super(RResNet, self).__init__()

        self.name = name

        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.linear = nn.Linear(512*block.expansion, num_classes)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)


    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def RResNet18(name='R-ResNet-18'):
    return RResNet(RBasicBlock, [2, 2, 2, 2], name=name)


def RResNet34(name='R-ResNet-34'):
    return RResNet(RBasicBlock, [3, 4, 6, 3], name=name)


def RResNet50(name='R-ResNet-50'):
    return RResNet(RBottleneck, [3, 4, 6, 3], name=name)


def RResNet101(name='R-ResNet-101'):
    return RResNet(RBottleneck, [3, 4, 23, 3], name=name)


def RResNet152(name='R-ResNet-152'):
    return RResNet(RBottleneck, [3, 8, 36, 3], name=name)
