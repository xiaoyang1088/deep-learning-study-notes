#学习最大池化如何缩小特征图，以及展平如何整理每个样本的特征
#对照局部最大值计算，观察NCHW到二维特征张量的形状变化

import torch
from torch import nn

image = torch.tensor([
    [1.0, 2.0, 3.0, 4.0],
    [5.0, 6.0, 7.0, 8.0],
    [9.0, 10.0, 11.0, 12.0],
    [13.0, 14.0, 15.0, 16.0]
    ],
    dtype=torch.float32
)

inputs=image.reshape(1,1,4,4)

pool_layer = nn.MaxPool2d(
    kernel_size=2,
    stride=2
)

flatten_layer=nn.Flatten(start_dim=1)

pooled_outputs=pool_layer(inputs)
flattened_outputs=flatten_layer(pooled_outputs)

print("输入图像：")
print(inputs[0, 0])
print("输入形状：", inputs.shape)

print("\n池化层：")
print(pool_layer)

print("\n池化结果：")
print(pooled_outputs[0, 0])
print("池化后形状：", pooled_outputs.shape)

print("\n展平结果：")
print(flattened_outputs)
print("展平后形状：", flattened_outputs.shape)

print("\n池化层参数：", list(pool_layer.parameters()))
print("展平层参数：", list(flatten_layer.parameters()))