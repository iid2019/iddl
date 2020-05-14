import torch
from thop import profile
import pickle
import sys
import copy
import matplotlib.pyplot as plt
sys.path.append('../bibd')
from bibd_layer import BibdLinear, RandomSparseLinear, generate_fake_bibd_mask, BibdConv2d, RandomSparseConv2d, bibd_sparsity
import numpy as np
from models.resnet import ResNet18, ResNet34, ResNet50, ResNet101
from models.resnet_bibd import BResNet18, BResNet34, BResNet50, BResNet101


MODEL_COUNT = 3 # The number of models of the same type. If there are 3 models of ResNet, then MODEL_COUNT = 3
SELECTED_INDEX_LIST = [0, 1, 2]
MODEL_INDEX_LIST = [0, 1, 2]

# Load the name and accuracy array
TIME = '20200513_062758'
accuracy_filename = 'accuracy_array_{}.pkl'.format(TIME)
accuracy_array = pickle.load(open(accuracy_filename, 'rb'))
print('accuracy_array:')
print(accuracy_array)
accuracy_ndarray = accuracy_array.reshape((3, MODEL_COUNT))
name_filename = 'model_name_array_{}.pkl'.format(TIME)
name_array = pickle.load(open(name_filename, 'rb'))

gpu_index = 0
if torch.cuda.is_available():
    device = torch.device("cuda:{}".format(gpu_index))
    print('CUDA available. PyTorch version:', torch.__version__, ' Device:', device)
else:
    print('CUDA is not available. Stopped.')
    sys.exit()


print('Building the models...')
model_list = []
model_list.append(ResNet18().to(device))
print('ResNet18 added.')
model_list.append(ResNet34().to(device))
print('ResNet34 added.')
model_list.append(ResNet50().to(device))
print('ResNet50 added.')
# model_list.append(ResNet101().to(device))
# print('ResNet101 added.')
model_list.append(BResNet18().to(device))
print('BResNet18 added.')
model_list.append(BResNet34().to(device))
print('BResNet34 added.')
model_list.append(BResNet50().to(device))
print('BResNet50 added.')
# model_list.append(BResNet101().to(device))
# print('BResNet101 added.')
print('All models built and added to the list.')
print('Model list:')
for model in model_list:
    print('    {}'.format(model.name))


def count_bibdConv2d(m, x: (torch.Tensor,), y: torch.Tensor):
    '''
    Reference: https://github.com/Lyken17/pytorch-OpCounter/blob/dbc052ec2d914e950b3e0fa24d3eea753a4e0bb7/thop/vision/basic_hooks.py#L15
    '''
    
    
    x = x[0]

    kernel_ops = torch.zeros(m.fpWeight.size()[2:]).numel()  # Kw x Kh
#     bias_ops = 1 if m.bias is not None else 0
    bias_ops = 0 # For our current implementation of BibdConv2d, bias is always disabled implicitly

    # N x Cout x H x W x  (Cin x Kw x Kh + bias)
    total_ops = y.nelement() * (m.in_channels // m.conGroups * kernel_ops + bias_ops)
    
    # Multiply total_ops with the sparsity
    total_ops *= bibd_sparsity(m.in_channels, m.out_channels)

    m.total_ops += torch.DoubleTensor([int(total_ops)])


custom_ops = {
    BibdConv2d: count_bibdConv2d,
    RandomSparseConv2d: count_bibdConv2d
}

# Calculate the FLOPs of all models
ops_array = np.array([], dtype=float)
for model in model_list:
    input = torch.randn(1, 3, 32, 32).to(device)
    flops, params = profile(model, inputs=(input, ), custom_ops=custom_ops)
    print('Model: %s, Params: %.4f, FLOPs(M): %.2f' % (model.name, params / (1000 ** 2), flops / (1000 ** 2)))
    ops_array = np.append(ops_array, flops / (1000 ** 2))
# Repeat the last MODEL_COUNT elements, because FLOPs of B-ResNet and R-ResNet should be the same
for i in range(MODEL_COUNT):
    ops = copy.deepcopy(ops_array[-MODEL_COUNT])
    ops_array = np.append(ops_array, ops)
# Reshape the array of ops
ops_ndarray = ops_array.reshape(3, MODEL_COUNT)

# Print all the data to be plotted
print('All the data to be plotted:')
name_list = [ 'ResNet', 'B-ResNet', 'R-ResNet' ]
print('name_list:')
print(name_list)
print('accuracy_ndarray:')
print(accuracy_ndarray)
print('ops_ndarray:')
print(ops_ndarray)

color_list = ['#22a7f0', '#cf000f', '#03a678']
shape_list = ['-o', '-^', '-s']

# Plot all models
fig, ax = plt.subplots(figsize=(10,6), dpi=200)
plt.setp(ax.get_xticklabels(), fontsize=18, fontweight="normal")
plt.setp(ax.get_yticklabels(), fontsize=18, fontweight="normal")
for i in MODEL_INDEX_LIST:
    ax.plot(ops_ndarray[i][SELECTED_INDEX_LIST], accuracy_ndarray[i][SELECTED_INDEX_LIST], shape_list[i], color=color_list[i], label=name_list[i])
ax.legend(fontsize=20)
ax.set_title(r'Accuracy v.s. FLOPs', fontsize=24)
ax.set_xlabel(r'FLOPs (M)', fontsize=20)
ax.set_ylabel(r'Accuracy', fontsize=20)

fig.savefig('fig_resnet_experiments.png', format='png', pad_inches=0)
fig.savefig('fig_resnet_experiments.eps', format='eps', pad_inches=0)

# Plot only sparse models: B-ResNet and R-ResNet
MODEL_INDEX_LIST = [1, 2]
fig, ax = plt.subplots(figsize=(10,6), dpi=200)
plt.setp(ax.get_xticklabels(), fontsize=18, fontweight="normal")
plt.setp(ax.get_yticklabels(), fontsize=18, fontweight="normal")
for i in MODEL_INDEX_LIST:
    ax.plot(ops_ndarray[i][SELECTED_INDEX_LIST], accuracy_ndarray[i][SELECTED_INDEX_LIST], shape_list[i], color=color_list[i], label=name_list[i])
ax.legend(fontsize=20)
ax.set_title(r'Accuracy v.s. FLOPs', fontsize=24)
ax.set_xlabel(r'FLOPs (M)', fontsize=20)
ax.set_ylabel(r'Accuracy', fontsize=20)

fig.savefig('fig_resnet_experiments_sparse_only.png', format='png', pad_inches=0)
fig.savefig('fig_resnet_experiments_sparse_only.eps', format='eps', pad_inches=0)

log_file = open('./log/plot_resnet_experiments_{}.log'.format(TIME), 'w')
