from sklearn.ensemble import AdaBoostClassifier
from skorch import NeuralNetClassifier
from models.resnet_bibd_v_ensemble import ResNet18
import torch
import torchvision
import torchvision.transforms as transforms


device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

train_data_list = []
train_labels_list = []

for batch_index, (inputs, targets) in enumerate(trainloader):
    # inputs, targets = inputs.to(device), targets.to(device)
    # print(batch_index)
    # print(inputs.shape)
    # print(targets.shape)

    train_data_list.append(inputs)
    train_labels_list.append(targets)

train_data = torch.cat(train_data_list, dim=0)
train_labels = torch.cat(train_labels_list, dim=0)
print(train_data.shape)
print(train_labels.shape)

# resnet18 = ResNet18()
# net = NeuralNetClassifier(module=resnet18)
# print(net)

# classifier = AdaBoostClassifier(base_estimator=net, n_estimators=20)
# print(classifier)

# classifier.fit(train_data, train_labels)
