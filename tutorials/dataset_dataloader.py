import torch
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
torch.manual_seed(1)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0), (1)), lambda x: x.view(-1)])

root = "./data"
mnist_train = datasets.MNIST(root = root, download = True, train = True, transform = transform)
mnist_test = datasets.MNIST(root = root, download = True, train = False, transform = transform)

train_dataloader = DataLoader(mnist_train, batch_size = 100, shuffle = True)
test_dataloader = DataLoader(mnist_train, batch_size = 100, shuffle = False)

print(f"mnist_train = {mnist_train}")
print(f"mnist_test = {mnist_test}")
print(f"train_dataloader = {train_dataloader}")
print(f"test_dataloader = {test_dataloader}")

x, t = next(iter(train_dataloader))
print(x.shape, t.shape)

#自作データセットの作成
class Dataset:
    def __init__(self):
        self.data = [1, 2, 3, 4, 5]
        self.label = [0, 0, 0, 1, 1]
    def __getitem__(self, index):
        return self.data[index], self.label[index]
    def __len__(self):
        return len(self.data)

dataset = Dataset()
print(f"dataset = {dataset}")
print(f"dataset[0] = {dataset[0]}, dataset[4] = {dataset[4]}")
print(f"len(dataset) = {len(dataset)}")

#datasetの前処理
class Dataset2:
    def __init__(self, transform_data = None, transform_label = None):
        self.transform_data = transform_data
        self.transform_label = transform_label
        self.data = [1, 2, 3, 4, 5, 6]
        self.label = [0, 0, 0, 1, 1, 1]
    def __getitem__(self, index):
        x = self.data[index]
        t = self.label[index]
        if self.transform_data:
            x = self.transform_data(self.data[index])
        if self.transform_label:
            x = self.transform_label(self.label[index])
        return x, t
    def __len__(self):
        return len(self.data)

#前処理含まない定義
dataset2 = Dataset2()
print(f"dataset2[0] = {dataset2[0]}, dataset2[4] = {dataset2[4]}")
print(f"len(dataset2) = {len(dataset2)}")

#前処理含む定義
transform = lambda x: x+10
dataset3 = Dataset2(transform_data = transform)
print(f"dataset3[0] = {dataset2[0]}, dataset3[4] = {dataset3[4]}")
print(f"len(dataset3) = {len(dataset3)}")

#Dataloaderに適応
dataloader = DataLoader(dataset3, batch_size=2, shuffle=True)
print(f"dataloader = {dataloader}")

for x, t in dataloader:
    print(f"x, t = {x}, {t}")
    print(f"x[0], x[1] = {x[0]}, {x[1]}")
    print(f"t[0], t[1] = {t[0]}, {t[1]}")