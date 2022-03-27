#原論文　https://arxiv.org/abs/2007.11824
#Ningning Ma, Xiangyu Zhang, Jian Sun "Funnel Activation for Visual Recognition", ECCV 2020

import torch
import torch.nn as nn

class FReLU(nn.Module):
    def __init__(self, inp, kernel=3, stride=1, padding=1):
        super().__init__()
        self.FC = nn.Conv2d(inp, inp, kernel_size=kernel, stride=stride, padding=padding, groups=inp)  #Depthwise畳み込み
        self.bn = nn.BatchNorm2d(inp)

    def forward(self, x):
        dx = self.bn(self.FC(x))
        return torch.max(x, dx)