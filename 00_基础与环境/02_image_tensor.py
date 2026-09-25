# 本文件使用三维张量模拟一张 RGB 图像。
# 用于理解通道、高度、宽度以及像素的含义。

import torch

image = torch.zeros(3,4,5)

print("image shape:",image.shape)
print("image ndim:",image.ndim)
print("number of values:",image.numel())
print("first channel shape:",image[0].shape)
print("pixel at row 1, column 2:",image[:,1,2])

image[0,1,2]=1.0
print("pixel after modification:",image[:,1,2])