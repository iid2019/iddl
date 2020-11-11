"""
Copyright 2019 - 2020, Jianfeng Hou, houjf@shanghaitech.edu.cn

All rights reserved.
"""

import torch
from thop import profile
import pickle
import sys
import copy
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
sys.path.append('../bibd')
from bibd_layer import BibdLinear, RandomSparseLinear, generate_fake_bibd_mask, BibdConv2d, RandomSparseConv2d, bibd_sparsity
import numpy as np
from models.sparse_resnet_v import create_resnet
from art import tprint
import collections
from matplotlib.ticker import FuncFormatter


tprint('IDDL', font='larry3d')

# Read the result dicts from the pickled files
print('All the data to be plotted:')
filename_list = [ 'ensemble_result_1_20200521_142555.pkl', 'ensemble_result_2_20200521_142607.pkl' ]
total_dict = {}
for filename in filename_list:
    with open(filename, "rb") as file:
        result_dict = pickle.load(file)
        for classifier_num in result_dict:
            total_dict[classifier_num] = result_dict[classifier_num]
ordered_total_dict = collections.OrderedDict(sorted(total_dict.items()))
classifier_num_list = []
accuracy_list = []
print('classifier_num, accuracy')
for classifier_num, accuracy in ordered_total_dict.items():
    classifier_num_list.append(classifier_num)
    accuracy_list.append(accuracy)
    print('{:d}, {:.4f}'.format(classifier_num, accuracy))


color_list = ['#7fc97f', '#beaed4', '#386cb0']
shape_list = ['-o', '-^', '-s']


def format_func(x, pos):
    return "{:.0f}%".format(x * 100)


def auto_label(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate("{:.2f}%".format(height * 100),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=18, fontweight="normal")


# Plot
fig, ax = plt.subplots(figsize=(10,6), dpi=200)
plt.setp(ax.get_xticklabels(), fontsize=18, fontweight="normal")
plt.setp(ax.get_yticklabels(), fontsize=18, fontweight="normal")
ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
ax.yaxis.set_major_formatter(FuncFormatter(format_func))
plt.xticks(classifier_num_list)
# ax.plot(classifier_num_list, accuracy_list, shape_list[0], color=color_list[0])
plt.ylim(0.73, 0.83)
rects = plt.bar(classifier_num_list, accuracy_list, color="#386cb0")
auto_label(rects)
rects[0].set_color("#beaed4")
# Plot the base line
plt.hlines(accuracy_list[0], 0, classifier_num_list[-1] + 1, colors='#f22613', linestyles='dashdot')
# ax.legend(fontsize=20)
# ax.set_title(r'# of classifiers v.s. accuracy', fontsize=24)
ax.set_xlabel(r'# of base classifiers', fontsize=20)
ax.set_ylabel(r'Accuracy', fontsize=20)

fig.savefig('fig_ensemble_experiments.png', format='png', pad_inches=0)
fig.savefig('fig_ensemble_experiments.eps', format='eps', pad_inches=0)

# # Plot improved accuracy
# improved_accuracy_list = []
# for accuracy in accuracy_list[1:]:
#     improved_accuracy_list.append(accuracy - accuracy_list[0])
# fig, ax = plt.subplots(figsize=(10,6), dpi=200)
# plt.setp(ax.get_xticklabels(), fontsize=18, fontweight="normal")
# plt.setp(ax.get_yticklabels(), fontsize=18, fontweight="normal")
# ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
# ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
# plt.xticks(classifier_num_list[1:])
# ax.plot(classifier_num_list[1:], improved_accuracy_list, shape_list[0], color=color_list[2])
# # ax.legend(fontsize=20)
# # ax.set_title(r'# of classifiers v.s. improved accuracy', fontsize=24)
# ax.set_xlabel(r'# of base classifiers', fontsize=20)
# ax.set_ylabel(r'Improved accuracy', fontsize=20)

# fig.savefig('fig_ensemble_experiments_improved.png', format='png', pad_inches=0)
# fig.savefig('fig_ensemble_experiments_improved.eps', format='eps', pad_inches=0)
