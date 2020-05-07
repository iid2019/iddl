import numpy as np
import matplotlib.pyplot as plt
import pickle
from models import *
import sys
sys.path.append('../bibd')
from bibd_layer import generate_fake_bibd_mask


TOTAL_NUM = 3
SELECTED_INDEX_LIST = [0, 1, 2]
MODEL_INDEX_LIST = [0, 1, 2]

numberOfParam_ndarray = np.zeros((3, TOTAL_NUM), dtype=float)
accuracy_ndarray = np.zeros((3, TOTAL_NUM), dtype=float)
name_list = ['MLP', 'B-MLP', 'R-MLP']

filename = 'mlp_experiments_{}.p'.format('20200507_150006')
experiment = pickle.load(open(filename, "rb"))

# print('epoch: 100')
for rowIndex, row in enumerate(experiment.acc_ndarray):
    name = experiment.model_name_array[rowIndex]
    accuracy = row[-1]

    modelIndex = rowIndex // TOTAL_NUM
    accuracy_ndarray[modelIndex][rowIndex % TOTAL_NUM] = accuracy
    print('{}: {}'.format(name, accuracy))

# print('epoch: 80')
# for rowIndex, row in enumerate(experiment.acc_ndarray):
#     print('{}: {}'.format(experiment.model_name_array[rowIndex], row[-21]))

# print('epoch: 30')
# for rowIndex, row in enumerate(experiment.acc_ndarray):
#     print('{}: {}'.format(experiment.model_name_array[rowIndex], row[-71]))


input_dim = 28*28*1
output_dim = 10

# layers_list = [layers2, layers3, layers4, layers5, layers7]
layers_list = [layers3_1, layers3_2, layers3_3]
def numberOfParam_mlp(layers):
    count = 0
    for index, dim in enumerate(layers):
        previous_dim = input_dim if index == 0 else layers[index-1]

        count += previous_dim * dim
    count += layers[-1] * output_dim
    return count

def numberOfParam_bibdMlp(layers):
    count = 0
    for index, dim in enumerate(layers):
        previous_dim = input_dim if index == 0 else layers[index-1]

        count += np.sum(generate_fake_bibd_mask(previous_dim, dim))
    count += layers[-1] * output_dim
    return count


for index, layers in enumerate(layers_list):
    numberOfParam_ndarray[0][index] = numberOfParam_mlp(layers)
for index, layers in enumerate(layers_list):
    numberOfParam_ndarray[1][index] = numberOfParam_bibdMlp(layers)
    numberOfParam_ndarray[2][index] = numberOfParam_bibdMlp(layers)

print(numberOfParam_ndarray)

color_list = ['#22a7f0', '#cf000f', '#03a678']
shape_list = ['-o', '-^', '-s']
fig, ax = plt.subplots(figsize=(10,6), dpi=200)
plt.setp(ax.get_xticklabels(), fontsize=18, fontweight="normal")
plt.setp(ax.get_yticklabels(), fontsize=18, fontweight="normal")
for i in MODEL_INDEX_LIST:
    ax.plot(numberOfParam_ndarray[i][SELECTED_INDEX_LIST], accuracy_ndarray[i][SELECTED_INDEX_LIST], shape_list[i], color=color_list[i], label=name_list[i])
ax.legend(fontsize=20)
ax.set_title(r'Accuracy v.s. # of parameters', fontsize=24)
ax.set_xlabel(r'# of parameters', fontsize=20)
ax.set_ylabel(r'Accuracy', fontsize=20)

fig.savefig('fig_mlp_experiments.png', format='png', pad_inches=0)
fig.savefig('fig_mlp_experiments.eps', format='eps', pad_inches=0)
