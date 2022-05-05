import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

df = pd.read_csv('./data/Concrete_Data.csv')    #データセット読み込み
print(f"最初の5行={df.head()}, 形式は{df.shape}")#その確認

# データを目的/説明変数に分類する(最終列だけ目的変数)
data = df.drop(df.columns[[-1]], axis=1)
target = df.iloc[:, -1]

# pytorch形式へ変換
data = torch.tensor(data.values, dtype=torch.float32)
target = torch.tensor(target.values, dtype=torch.float32)

dataset = torch.utils.data.TensorDataset(data, target)

# サンプル数を設定
n_train = int(len(dataset)*0.6)
n_val = int((len(dataset)-n_train)*0.5)
n_test = len(dataset)-n_train-n_val

torch.manual_seed(0)
train, val, test = torch.utils.data.random_split(dataset, [n_train, n_val, n_test])

batch_size = 32

torch.manual_seed(0)

# shuffle はデフォルトで False のため、学習データのみ True に指定
train_loader = torch.utils.data.DataLoader(train, batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val, batch_size)
test_loader = torch.utils.data.DataLoader(test, batch_size)

# 辞書型変数にまとめる(trainとvalをまとめて出す)
dataloaders_dict = {"train": train_loader, "val": val_loader}

class Net(nn.Module):

    # 使用するオブジェクトを定義
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(8, 4)
        self.fc2 = nn.Linear(4, 3)
        self.fc3 = nn.Linear(3, 1)

    # 順伝播
    def forward(self, x):
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = x**2*+1
        x = self.fc3(x)
        x = F.relu(x)        
        
        return x    

# インスタンス化
net = Net()

# ネットワークの確認
net

# 損失関数の設定(最小二乗誤差)
criterion = nn.MSELoss()

# 最適化手法の選択
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

def train_model(net, dataloader_dict, lossfun, optimizer, num_epoch):
    
    l= []
    
    # 重みを保持する変数
    best_acc = 0.0

    # GPUが使えるのであればGPUを有効化する
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    
    # (エポック)回分のループ
    for epoch in range(num_epoch):

        
        for phase in ['train', 'val']:
            
            if phase == 'train':
                # 学習モード
                net.train()
            else:
                # 推論モード
                net.eval()
                
            epoch_loss = 0.0
            
            # 第1回で作成したDataLoaderを使ってデータを読み込む
            for inputs, labels in tqdm(dataloaders_dict[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                labels = torch.reshape(labels,(-1,1)) #サイズ変更
                # 勾配を初期化する
                optimizer.zero_grad()
                
                # 学習モードの場合のみ勾配の計算を可能にする
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(inputs)
                    _, preds = torch.max(outputs, 1)
                    # 損失関数を使って損失を計算する
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        # 誤差を逆伝搬する
                        loss.backward()
                        # パラメータを更新する
                        optimizer.step()
                        
                    epoch_loss += loss.item() * inputs.size(0)
                    
            # 1エポックでの損失を計算
            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            
            #lossをデータで保存する
            a_loss = np.array(epoch_loss)
            l.append(a_loss)        
        
        #epoch数とlossを表示する
        print('Epoch {}/{}'.format(epoch + 1, num_epoch))   
        print('epoch_loss:{:.4f}'.format(epoch_loss))
        print('-'*20) 
        
        #モデルを保存
        torch.save(net, 'best_model.pth')
                
    #testとvalのlossとaccを抜き出してデータフレーム化
    l_train = l[::2]
    l_train = pd.DataFrame({'train_loss_kaiki':l_train})
            
    l_val = l[1::2]
    l_val = pd.DataFrame({'val_loss_kaiki':l_val})
                                   
    df_loss = pd.concat((l_train,l_val),axis=1)    
    
    #ループ終了後にdfを保存
    df_loss.to_csv('./data/loss_kaiki.csv', encoding='shift_jis')
               
          
#学習と検証
num_epoch = 100
net = train_model(net, dataloaders_dict, criterion, optimizer, num_epoch)