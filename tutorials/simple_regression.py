import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
torch.manual_seed(1)

x = torch.normal(5, 1, size = (10,))
print(f"x = {x}")
t = 3*x + 2 + torch.randn(10)
print(f"t = {t}")
x_train = x.unsqueeze(1).float()
print(f"x_train = {x_train}")
t_train = t.unsqueeze(1).float()
print(f"t_train = {t_train}")

#plt.scatter(x_train, t_train)
#plt.show()

model = nn.Linear(1, 1)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr = 1e-2)

#パラメータを表示
print(list(model.parameters()))

for i in range(1, 6):
    y = model(x_train)
    loss_train = criterion(y, t_train)
    #勾配を初期化
    optimizer.zero_grad()
    #自動微分
    loss_train.backward()
    #最適化処理
    optimizer.step()
    print(f"{i}回目で{loss_train}の誤差")
    print(f"この時、a = {model.weight}, b = {model.bias}")

x_result = torch.arange(3, 7).unsqueeze(1).float()
y = model(x_result)
#求めた漸近線をプロット
plt.plot(x_result, y.detach())
#訓練データを散布図として表示
plt.scatter(x_train, t_train)
plt.show()