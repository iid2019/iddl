'''
Copyright 2019 - 2020, Jianfeng Hou, houjf@shanghaitech.edu.cn

All rights reserved.

ResNet for CIFAR-10.
Reference: https://github.com/kuangliu/pytorch-cifar
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1


    def __init__(self, in_planes, planes, num_partition, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, groups=num_partition)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False, groups=num_partition)
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


class ResNet(nn.Module):
    # The number of partitions
    # num_partition = 4
    num_partition = 8
    
    
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], num_partition=self.num_partition, stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], num_partition=self.num_partition, stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], num_partition=self.num_partition, stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], num_partition=self.num_partition, stride=2)

        # self.linear_list = []
        # for i in range(self.num_partition):
        #     self.linear_list.append(nn.Linear(int(512/self.num_partition), num_classes))
        # print(self.linear_list)
        # TODO
        self.linear1 = nn.Linear(int(512/self.num_partition), num_classes)
        self.linear2 = nn.Linear(int(512/self.num_partition), num_classes)
        self.linear3 = nn.Linear(int(512/self.num_partition), num_classes)
        self.linear4 = nn.Linear(int(512/self.num_partition), num_classes)

        self.linear = nn.Linear(self.num_partition * num_classes, num_classes)


    def _make_layer(self, block, planes, num_blocks, num_partition, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, num_partition, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)


    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)

        # print(out)
        # print(out.size())

        # Partition
        partition_list = []
        for partition_index in range(self.num_partition):
            partition_length = int(out.size(1) / self.num_partition)
            # print('partition_length = {}'.format(partition_length))
            partition = torch.narrow(out, 1, partition_length * (partition_index), partition_length)
            # print('partition.size(): {}'.format(partition.size()))
            # print('partition = {}'.format(partition))
            partition_list.append(partition)

        # Calculate the output of different classifiers (partitions)
        output_list = []
        # for i, partition in enumerate(partition_list):
        #     output_list.append(self.linear_list[i](partition))
        output_list.append(self.linear1(partition_list[0]))
        output_list.append(self.linear2(partition_list[1]))
        output_list.append(self.linear3(partition_list[2]))
        output_list.append(self.linear4(partition_list[3]))

        out = self.linear(torch.cat(output_list, 1))

        return out


def EResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


# # Test
# net = ResNet18()
# print(net)
# y = net(torch.randn(1, 3, 32, 32))
# print(y.size())
