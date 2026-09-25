# 本文件创建二维张量，并查看形状、维度、数据类型和所在设备。
# 用于认识 PyTorch 张量的基本属性。

import torch

matrix = torch.tensor([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0]
])

print("matrix:")
print(matrix)
print("shape:", matrix.shape)
print("ndim:", matrix.ndim)
print("dtype:", matrix.dtype)
print("device:", matrix.device)
