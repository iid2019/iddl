import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from models import *
from experiment import Experiment
import pickle
from datetime import datetime


print('MLP experiments started')

# Use start time for the filename of the pickled file
date_time = datetime.now().strftime("%Y%m%d_%H%M%S")


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
print('Using PyTorch version:', torch.__version__, ' Device:', device)


batch_size = 32

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

validation_dataset = datasets.MNIST('./data', train=False, transform=transforms.ToTensor())
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=True)


input_dim = 28 * 28 * 1
output_dim = 10

model_list = []
if torch.cuda.is_available():
    model_list.append(Mlp2(input_dim, output_dim).to(device))
    model_list.append(Mlp3(input_dim, output_dim).to(device))
    model_list.append(Mlp4(input_dim, output_dim).to(device))
    model_list.append(Mlp5(input_dim, output_dim).to(device))
    model_list.append(Mlp7(input_dim, output_dim).to(device))
    model_list.append(BibdMlp2(input_dim, output_dim).to(device))
    model_list.append(BibdMlp3(input_dim, output_dim).to(device))
    model_list.append(BibdMlp4(input_dim, output_dim).to(device))
    model_list.append(BibdMlp5(input_dim, output_dim).to(device))
    model_list.append(BibdMlp7(input_dim, output_dim).to(device))
    model_list.append(RandomSparseMlp2(input_dim, output_dim).to(device))
    model_list.append(RandomSparseMlp3(input_dim, output_dim).to(device))
    model_list.append(RandomSparseMlp4(input_dim, output_dim).to(device))
    model_list.append(RandomSparseMlp5(input_dim, output_dim).to(device))
    model_list.append(RandomSparseMlp7(input_dim, output_dim).to(device))
else:
    print('CUDA is not available. Stopped.')
print('model_list: ')
for model in model_list:
    print('   {}'.format(model.name))


experiment = Experiment(n_epoch=30)
for model in model_list:
    experiment.run_model(model, train_loader, validation_loader)


# Save all the experiment data
filename = 'mlp_experiments_{}.p'.format(date_time)
pickle.dump(experiment, open(filename, "wb"))
print('The Experiment instance experiment dumped to the file: {}'.format(filename))

print('MLP experiments completed at {}'.format(datetime.now().strftime("%Y%m%d_%H%M%S")))
