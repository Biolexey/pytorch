from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import matplotlib.pyplot as plt
torch.manual_seed(1)

#ベクトルに直す前処理(.view(-1))を削除
#transformについてはhttps://pystyle.info/pytorch-list-of-transforms/ を参照
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0), (1))])

#rootディレクトリを指定してMNIST画像ファイルをそこに保存
root = "./data"
mnist_train = datasets.MNIST(root = root, download = True, train = True, transform = transform)
mnist_test = datasets.MNIST(root = root, download = True, train = False, transform = transform)

#dataloaderに画像ファイルを設定する(バッチサイズは100に設定)
train_dataloader = DataLoader(mnist_train, batch_size = 100, shuffle = True)
test_dataloader = DataLoader(mnist_train, batch_size = 100, shuffle = False)

#dataloaderが機能しているか画像を表示して確認
x, t = next(iter(train_dataloader))
image = x[0,].view(28, 28).detach().numpy()
plt.imshow(image, cmap = "binary_r")
plt.show()

#gglcolabの場合はGPU駆動に設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)#確認

class Model(nn.Module):
    #Modelを定義
    def __init__(self):
        super().__init__()
        #(input channels, output channels, kernel, stride, padding)
        self.c1 = nn.Conv2d(1, 32, kernel_size = 4, stride = 2, padding = 1)#28*28→14*14
        self.b1 = nn.BatchNorm2d(32)
        self.c2 = nn.Conv2d(32, 64, kernel_size = 4, stride = 2, padding = 1)#14*14→7*7
        self.b2 = nn.BatchNorm2d(64)
        self.l1 = nn.Linear(576, 128)
        self.b3 = nn.BatchNorm1d(128)
        self.l2 = nn.Linear(128, 10)
    #forward計算式
    def forward(self, x):
        x = torch.relu(self.b1(self.c1(x)))
        x = F.max_pool2d(torch.relu(self.b2(self.c2(x))), 2)#7*7→3*3
        #ここでベクトルに変換
        x = x.view(-1, 576)#576=3*3*64
        x = torch.relu(self.b3(self.l1(x)))
        x = self.l2(x)
        return x

model = Model().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = 1e-2)

loss_train_all = []
acc_train_all = []
loss_test_all = []
acc_test_all = []

for epoch in range(1, 20+1):
    loss_train = 0
    acc_train = 0
    loss_test = 0
    acc_test = 0

    model.train()
    for x, t in train_dataloader:
        x, t = x.to(device), t.to(device)
        y = model(x)
        loss = criterion(y, t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_train += loss.item()
        acc_train += sum(y.argmax(axis = 1) == t) / len(t)
    
    loss_train_mean = loss_train / len(train_dataloader)
    acc_train_mean = acc_train / len(train_dataloader)

    model.eval()
    with torch.no_grad():
        for x, t in train_dataloader:
            x, t = x.to(device), t.to(device)
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

    if epoch == 1 or epoch % 5 == 0:
        print(f"Epoch : {epoch}")
        print(f"loss_train : {loss_train_mean: .4f}, acc_train : {acc_train_mean: .4f}")
        print(f"loss_test : {loss_test_mean: .4f}, acc_test : {acc_test_mean: .4f}")

fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

ax1.plot(range(1, len(loss_train_all)+1), loss_train_all, label = "train")
ax1.plot(range(1, len(loss_test_all)+1), loss_test_all, label = "test")
ax1.legend()

ax2.plot(range(1, len(acc_train_all)+1), acc_train_all, label = "train")
ax2.plot(range(1, len(acc_test_all)+1), acc_test_all, label = "test")
ax2.legend()
plt.show()