import torch
import numpy as np 

#データから直接テンソルに変換
data = [[1,2],[3,4]]
x_data = torch.tensor(data)
print(x_data)

#Numpyarrayからテンソルに変換
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
print(x_np)

#他のテンソルから作成
x_ones = torch.ones_like(x_data) # x_dataの特性（プロパティ）を維持
print(f"Ones Tensor: \n {x_ones} \n")
x_rand = torch.rand_like(x_data, dtype=torch.float) # x_dataのdatatypeを上書き更新
print(f"Random Tensor: \n {x_rand} \n")

#ランダム値や定数のテンソルの作成
shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)
print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")

#テンソルの属性変数
tensor = torch.rand(3,4)
print(tensor)
print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

# GPUが使用可能であれば、GPU上にテンソルを移動させる
if torch.cuda.is_available():
    print("yes")
    tensor = tensor.to('cuda')

#numpy-likeなindexingとslicing:
tensor = torch.ones(4, 4)
print('First row: ',tensor[0])
print('First column: ', tensor[:, 0])
print('First column: ', tensor[:, 0:1])#slicingだと次元が保存される
print('Last column:', tensor[..., -1])
tensor[:,1] = 7
print(tensor)

#テンソルの結合
t1 = torch.cat([tensor, tensor, tensor], dim=1)
t2 = torch.cat([tensor, tensor, tensor], dim=-1)
a = True
for i in range(4):
    for j in range(12):
        if t1[i,j] != t2[i,j]:
            a = False
if a:    
    print("same")
    print(t1)

# 2つのテンソル行列のかけ算です。 y1, y2, y3 は同じ結果になります。
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)
y3 = torch.rand_like(tensor)
torch.matmul(tensor, tensor.T, out=y3)
print("y1= {}".format(y1))

# こちらは、要素ごとの積を求めます。 z1, z2, z3 は同じ値になります。
z1 = tensor * tensor
z2 = tensor.mul(tensor)
z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)
print("z1= {}".format(z1))

#　1要素のテンソル
agg = tensor.sum()
agg_item = agg.item() #item()でpython型に変換
print(agg, type(agg))
print(agg_item, type(agg_item))

#インプレース操作(オペランドを直に変更)
print(tensor, "\n")
tensor.add_(5)
print(tensor)
print(tensor.copy_(z1))
print(tensor.t_())

#numpyへの変換
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

t.add_(1)
print(f"t: {t}")
print(f"n: {n}")

n = np.ones(5)
t = torch.from_numpy(n)

np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")

#test