from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
torch.manual_seed(1)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0), (1)), lambda x: x.view(-1)])

root = "./data"
mnist_train = datasets.MNIST(root = root, download = True, train = True, transform = transform)
mnist_test = datasets.MNIST(root = root, download = True, train = False, transform = transform)

train_dataloader = DataLoader(mnist_train, batch_size = 100, shuffle = True)
test_dataloader = DataLoader(mnist_train, batch_size = 100, shuffle = False)

x, t = next(iter(train_dataloader))
image = x[0,].view(28, 28).detach().numpy()
plt.imshow(image, cmap = "binary_r")
plt.show()

model = nn.Sequential(nn.Linear(784, 200), nn.ReLU(), nn.Linear(200, 10))
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = 1e-2)

loss_train_all = []
acc_train_all = []
loss_test_all = []
acc_test_all = []

for epoch in range(1, 50+1):
    loss_train = 0
    acc_train = 0
    loss_test = 0
    acc_test = 0

    for x, t in train_dataloader:
        y = model(x)
        loss = criterion(y, t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_train += loss.item()
        acc_train += sum(y.argmax(axis = 1) == t) / len(t)
    
    loss_train_mean = loss_train / len(train_dataloader)
    acc_train_mean = acc_train / len(train_dataloader)

    with torch.no_grad():
        for x, t in train_dataloader:
            y = model(x)
            loss = criterion(y, t)

            loss_test += loss.item()
            acc_test += sum(y.argmax(axis = 1) == t) / len(t)
    
    loss_test_mean = loss_test / len(test_dataloader)
    acc_test_mean = acc_test / len(test_dataloader)

    loss_train_all.append(loss_train_mean)
    acc_train_all.append(acc_train_mean)
    loss_test_all.append(loss_test_mean)
    acc_test_all.append(acc_test_mean)

    if epoch == 1 or epoch % 10 == 0:
        print(f"Epoch : {epoch}")
        print(f"loss_train : {loss_train_mean: .4f}, acc_train : {acc_train_mean: .4f}")
        print(f"loss_test : {loss_test_mean: .4f}, acc_test : {acc_test_mean: .4f}")