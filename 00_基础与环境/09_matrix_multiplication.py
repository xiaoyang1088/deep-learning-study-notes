#本文件比较张量的逐元素乘法和矩阵乘法
#用于同时通过形状改变理解神经网络线性计算的基本形式

import torch

matrix_a=torch.tensor([
    [1.0,2.0],
    [3.0,4.0]
])

matrix_b=torch.tensor([
    [5.0,6.0],
    [7.0,8.0]
])

elementwise_result=matrix_a*matrix_b
matrix_result=matrix_a@matrix_b

print("matrix A:",matrix_a)
print("matrix B:",matrix_b)
print("elementwise_result:",elementwise_result)
print("matrix_result:",matrix_result)

features=torch.tensor([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0]
])

weights = torch.tensor([
    [1.0, 0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0, 1.0],
    [1.0, 1.0, 1.0, 1.0],
])

outputs = features @ weights

print("features shape:", features.shape)
print("weights shape:", weights.shape)
print("outputs shape:", outputs.shape)
print("outputs:",outputs)