#使用固定的二维卷积核处理一张小型单通道图像
#对照手工计算，并观察步幅和填充对输出形状的影响

import torch 
from torch import nn


image= torch.tensor([
    [1.0, 2.0, 3.0, 4.0],
    [5.0, 6.0, 7.0, 8.0],
    [9.0, 10.0, 11.0, 12.0],
    [13.0, 14.0, 15.0, 16.0]
],
dtype=torch.float32
)


inputs=image.reshape(1,1,4,4)

kernel=torch.tensor([
    [1.0,0.0],
    [0.0,1.0]
],dtype=torch.float32)

conv_layer=nn.Conv2d(
    in_channels=1,
    out_channels=1,
    kernel_size=2,
    stride=1,
    padding=0,
    bias=False
)

with torch.no_grad():
    conv_layer.weight.copy_(kernel.reshape(1,1,2,2))
    outputs=conv_layer(inputs)

print("原始图像：")
print(image)

print("\n输入形状：", inputs.shape)

print("\n卷积核：")
print(conv_layer.weight[0, 0])
print("权重形状：", conv_layer.weight.shape)

print("\n输出形状：", outputs.shape)
print("输出特征图：")
print(outputs[0, 0])