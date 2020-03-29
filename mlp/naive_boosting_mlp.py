"""
Reference: https://blog.csdn.net/york1996/article/details/83111197
"""

import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import Counter
import numpy as np
 
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.input_layer=nn.Sequential(
            nn.Linear(28*28,30),
            nn.Tanh(),
        )
        self.output_layer=nn.Sequential(
            nn.Linear(30,10),
            #nn.Sigmoid()
        )
    def forward(self, x):
        x=x.view(x.size(0),-1)
        x=self.input_layer(x)
        x=self.output_layer(x)
        return x
 
trans=torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([.5],[.5]),
    ]
)
BATCHSIZE=100
EPOCHES=200
LR=0.001
 
train_data=torchvision.datasets.MNIST(root="./data", train=True, transform=trans, download=True)
test_data=torchvision.datasets.MNIST(root="./data", train=False, transform=trans, download=True)
train_loader=DataLoader(train_data,batch_size=BATCHSIZE,shuffle=True)
test_loader =DataLoader(test_data,batch_size=BATCHSIZE,shuffle=False)
 
mlps=[MLP().cuda() for i in range(10)]
optimizer=torch.optim.Adam([{"params":mlp.parameters()} for mlp in mlps],lr=LR)
 
 
loss_function=nn.CrossEntropyLoss()
 
for ep in range(EPOCHES):
    for img,label in train_loader:
        img,label=img.cuda(),label.cuda()
        optimizer.zero_grad()#10个网络清除梯度
        for mlp in mlps:
            out=mlp(img)
            loss=loss_function(out,label)
            loss.backward()#网络们获得梯度
        optimizer.step()
 
    pre=[]
    vote_correct=0
    mlps_correct=[0 for i in range(len(mlps))]
    for img,label in test_loader:
        img,label=img.cuda(),label.cuda()
        for i, mlp in  enumerate( mlps):
            out=mlp(img)
 
            _,prediction=torch.max(out,1) #按行取最大值
            pre_num=prediction.cpu().numpy()
            mlps_correct[i]+=(pre_num==label.cpu().numpy()).sum()
 
            pre.append(pre_num)
        arr=np.array(pre)
        pre.clear()
        result=[Counter(arr[:,i]).most_common(1)[0][0] for i in range(BATCHSIZE)]
        vote_correct+=(result == label.cpu().numpy()).sum()
    print("epoch:" + str(ep)+"总的正确率"+str(vote_correct/len(test_data)))
 
    for idx, coreect in enumerate(mlps_correct):
        print("网络"+str(idx)+"的正确率为："+str(coreect/len(test_data)))
