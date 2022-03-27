import torch
import torchvision
import numpy
from torchvision import transforms 

import torch.nn as nn

import torch.optim as optim

import matplotlib.pyplot as plt
"""
import sys
sys.path.append("..")
from functions.FReLU import FReLU
"""

class FReLU(nn.Module):
    def __init__(self, inp, kernel=3, stride=1, padding=1):
        super().__init__()
        self.FC = nn.Conv2d(inp, inp, kernel_size=kernel, stride=stride, padding=padding, groups=inp)  #Depthwise畳み込み
        self.bn = nn.BatchNorm2d(inp)

    def forward(self, x):
        dx = self.bn(self.FC(x))
        return torch.max(x, dx)

EPOCH = 20
BATCH_SIZE = 100

#データの前処理
#テンソル化と正規化を行う
trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0), (1))])

#データのダウンロード
trainset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=trans)
#print(trainset[0])

#dataloaderの設定
#num_workersは並列処理の数。winでは2以上だとエラーになる可能性あり。
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=0)
#print(trainloader)
#dataloaderはiter型なので直接参照する場合は以下のようにする必要がある。(この確認は学習前に行ってはいけない。iterなので一度取り出したら二週目まで取り出せなくなる。)
"""
for data, label in trainloader:
    break
print(label)
"""
#テストデータも同様に取得(シャッフルは無効)
testset = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=trans)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)

#モデルの定義
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.frelu16 = FReLU(16)
        self.frelu32 = FReLU(32)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, stride=2)

        self.conv1 = nn.Conv2d(1, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)

        self.fc1 = nn.Linear(32*5*5, 120)
        self.fc2 = nn.Linear(120, 10)

    def forward(self, x):
        x = self.conv1(x)#28→26
        x = self.frelu16(x)
        x = self.pool(x)#26→13
        x = self.conv2(x)#13→11
        x = self.frelu32(x)
        x = self.pool(x)#11→5
        x = x.view(x.size()[0], -1)#ここでベクトルに直す
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

#モデルの準備と損失関数と最適化手法の設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = Net().to(device)
criterion = nn.CrossEntropyLoss()#損失関数はクロスエントロピー誤差
#最適化は確率的勾配降下法
#lrは学習率、momentumは慣性項(大きいほど更新量大)、weight_decayは正則化項(大きいほど過学習抑制)
optimizer = optim.SGD(net.parameters(), lr=1e-2, momentum=0.9, weight_decay=0.005)

train_loss_value=[]      #trainingのlossを保持するlist
train_acc_value=[]       #trainingのaccuracyを保持するlist
test_loss_value=[]       #testのlossを保持するlist
test_acc_value=[]        #testのaccuracyを保持するlist 

#学習
for epoch in range(EPOCH):#epoch=学習回数=20
        
    sum_loss = 0.0
    sum_correct = 0
    sum_total = 0

    #トレーニングデータ
    for inputs, labels in trainloader:#全データ60000/バッチサイズ100=600回
        inputs, labels = inputs.to(device), labels.to(device)
        #勾配情報を初期化
        optimizer.zero_grad()
        outputs = net(inputs)
        #outputsには各ラベルである確率が10要素のテンソルとして格納されてるが、CrossEntropyLoss()にはSoftmaxが含まれているのでそのまま代入で問題ない。
        loss = criterion(outputs, labels)
        sum_loss += loss.item()                            #lossを足していく
        _, predicted = outputs.max(1)                      #出力の最大値の添字(予想位置)を取得
        sum_total += labels.size(0)                        #labelの数を足していくことでデータの総和を取る
        sum_correct += (predicted == labels).sum().item()  #予想位置と実際の正解を比べ,正解している数だけ足す
        loss.backward()
        optimizer.step()
    print("epoch={}, train mean loss={}, accuracy={}"
        .format(epoch, sum_loss*BATCH_SIZE/len(trainloader.dataset), float(sum_correct/sum_total)))  #lossとaccuracy出力
    train_loss_value.append(sum_loss*BATCH_SIZE/len(trainloader.dataset))  #traindataのlossをグラフ描画のためにlistに保持
    train_acc_value.append(float(sum_correct/sum_total))   #traindataのaccuracyをグラフ描画のためにlistに保持

    #テストデータ
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        sum_loss += loss.item()
        _, predicted = outputs.max(1)
        sum_total += labels.size(0)
        sum_correct += (predicted == labels).sum().item()
    print("epoch={}, test  mean loss={}, accuracy={}"
        .format(epoch, sum_loss*BATCH_SIZE/len(testloader.dataset), float(sum_correct/sum_total)))
    test_loss_value.append(sum_loss*BATCH_SIZE/len(testloader.dataset))
    test_acc_value.append(float(sum_correct/sum_total))




plt.figure(figsize=(6,6))      #グラフ描画用

#以下グラフ描画
plt.plot(range(EPOCH), train_loss_value)
plt.plot(range(EPOCH), test_loss_value, c='#00ff00')
plt.xlim(0, EPOCH)
plt.ylim(0, 2.5)
plt.xlabel('EPOCH')
plt.ylabel('LOSS')
plt.legend(['train loss', 'test loss'])
plt.title('loss')
plt.savefig("loss_image.png")
plt.clf()

plt.plot(range(EPOCH), train_acc_value)
plt.plot(range(EPOCH), test_acc_value, c='#00ff00')
plt.xlim(0, EPOCH)
plt.ylim(0, 1)
plt.xlabel('EPOCH')
plt.ylabel('ACCURACY')
plt.legend(['train acc', 'test acc'])
plt.title('accuracy')
plt.savefig("accuracy_image.png")