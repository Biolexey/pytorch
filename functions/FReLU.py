#原論文　https://arxiv.org/abs/2007.11824
#Ningning Ma, Xiangyu Zhang, Jian Sun "Funnel Activation for Visual Recognition", ECCV 2020

import torch
import torch.nn as nn

class FReLU(nn.Module):
    def __init__(self, input, kernel=3, stride=1, padding=1):
        super().__init__()
        self.FC = nn.conv2d(input, input, kernel=kernel, stride=stride, padding=padding, groups=input)  #Depthwise畳み込み
        self.bn = nn.BatchNorm2d(input)

    def forward(self, x):
        dx = self.bn(self.FC(x))
        return torch.max(x, dx)