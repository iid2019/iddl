"""Sparse ResNet with vertical partitioning in PyTorch.

Author: Jianfeng Hou, houjf@shanghaitech.edu.cn

The sparsification technique is applied to convolution layers.

The vertical partitioning technique is applied via group convolution.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('../bibd')
from bibd_layer import BibdConv2d, RandomSparseConv2d


class BasicBlock(nn.Module):
    expansion = 1


    def __init__(self, in_planes, planes, conv_layer, stride=1, num_groups=1):
        super(BasicBlock, self).__init__()

        self.conv1 = conv_layer(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, groups=num_groups)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv_layer(planes, planes, kernel_size=3, stride=1, padding=1, bias=False, groups=num_groups)
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


class Bottleneck(nn.Module):
    expansion = 4


    def __init__(self, in_planes, planes, conv_layer, stride=1, num_groups=1):
        super(Bottleneck, self).__init__()

        self.conv1 = conv_layer(in_planes, planes, kernel_size=1, bias=False, groups=num_groups)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv_layer(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, groups=num_groups)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv_layer(planes, self.expansion*planes, kernel_size=1, bias=False, groups=num_groups)
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


class SparseResNetV(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, sparsification='none', num_groups=1, name='SparseResNetV'):
        '''Constructor for the SparseResNetV class.

        Args:
            block: The building block.
            num_bloks: List[int]. The list of number of blocks.
            num_classes: int. The number of classes for the classification problem.
            sparsification: str. The sparsification technique that should be used.
                'none': no sparsification, 'bibd': BIBD sparsification, 'random': random sparsification.
            name: str. The name of this model.
        '''


        super(SparseResNetV, self).__init__()

        # Choose conv_layer according to the sparsification technique
        if sparsification is 'none':
            conv_layer = nn.Conv2d
        elif sparsification is 'bibd':
            conv_layer = BibdConv2d
        elif sparsification is 'random':
            conv_layer = RandomSparseConv2d
        else:
            raise ValueError("The parameter sparsification must be one of ['none', 'bibd', 'random'].")

        self.name = name

        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], 1, conv_layer, num_groups=num_groups)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], 2, conv_layer, num_groups=num_groups)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], 2, conv_layer, num_groups=num_groups)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], 2, conv_layer, num_groups=num_groups)

        self.linear = nn.Linear(512 * block.expansion, num_classes)


    def _make_layer(self, block, planes, num_blocks, stride, conv_layer, num_groups=1):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, conv_layer, stride=stride, num_groups=num_groups))
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
        out = self.linear(out)

        return out


PARAM_DICT = {
    '18': (BasicBlock, [2, 2, 2, 2], 'SparseResNet-18-V'),
    '34': (BasicBlock, [3, 4, 6, 3], 'SparseResNet-34-V'),
    '50': (Bottleneck, [3, 4, 6, 3], 'SparseResNet-50-V'),
    '101': (Bottleneck, [3, 4, 23, 3], 'SparseResNet-101-V'),
    '152': (Bottleneck, [3, 8, 36, 3], 'SparseResNet-152-V')
}


def create_resnet(arch: str, sparsification='none', num_groups=1, name=None):
    """Creates an instance of the class SparseResNetV with the specified parameters.

    Args:
        arch: The base architecture of the ResNet to create.
            Legal values are [ '18', '34', '50', '101', '152' ], for ResNet-18, ResNet-34, ResNet-50, ResNet-101, ResNet-152 respectively.
        sparsification: str. The sparsification technique that should be used.
            'none': no sparsification, 'bibd': BIBD sparsification, 'random': random sparsification.
        num_groups: int. The number of groups for vertical partitioning. Should be a positive integer.
        name: str. The name of the model.
    """

    # Check arch parameter
    if arch not in PARAM_DICT:
        raise ValueError("The parameter arch must be one of [ '18', '34', '50', '101', '152' ].")

    params = PARAM_DICT[arch]
    return SparseResNetV(
        params[0],
        params[1],
        sparsification=sparsification,
        num_groups=num_groups,
        name=params[2] if name is None else name)
