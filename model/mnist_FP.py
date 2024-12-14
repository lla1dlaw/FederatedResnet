import torch
import torch.nn as nn
import torchvision.transforms as transforms
import math
import matplotlib.pyplot as plt
import torchvision
import math
__all__ = ['mnist_FP']

class mnistNet(nn.Module):
    def __init__(self):
        super(mnistNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.tanh1 = nn.Tanh()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.tanh2 = nn.Tanh()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(7*7*16, 100)
        self.bn3 = nn.BatchNorm1d(100)
        self.tanh3 = nn.Tanh()
        self.fc2 = nn.Linear(100, 10)
        self.bn4 = nn.BatchNorm1d(10)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.original_weights = {}


    def forward(self, x):
        
        if self.training:
            self.original_weights = {
                "conv1": self.conv1.weight.data.clone(),
                "conv2": self.conv2.weight.data.clone(),
                "fc1": self.fc1.weight.data.clone(),
                "fc2": self.fc2.weight.data.clone(),
            }
        x = x.view(-1, 1, 28, 28)
        x = self.maxpool1(self.tanh1(self.bn1(self.conv1(x))))
        x = self.maxpool2(self.tanh2(self.bn2(self.conv2(x))))
        x = x.view(-1, 7*7*16)
        x = self.tanh3(self.bn3(self.fc1(x)))
        x = self.logsoftmax(self.bn4(self.fc2(x)))

        return x,self.original_weights
    
def mnist_FP(**model_config):

    return mnistNet()
