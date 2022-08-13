import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms
import numpy as np
import matplotlib.pyplot as plt

G = torch.load("./Generator_epoch_29.pth")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for i in input():
  print("real")
  plt.imshow(i[0][0].reshape(28,28))
  plt.show()
  real_inputs = i[0][0]
  noise = (torch.rand(real_inputs.shape[0], 128)-0.5)/0.5
  noise = noise.to(device)
  fake_inputs = G(noise)
  print("fake")
  plt.imshow(fake_inputs[0][0].cpu().detach().numpy().reshape(28,28))
  plt.show()
  break