#本文件演示ReLU如何在两个线性层之间加入非线性
#通过固定权重，让简单网络计算输入数字的绝对值

import torch
from torch import nn

inputs = torch.tensor([
    [-2.0],
    [-1.0],
    [0.0],
    [1.0],
    [2.0],
])

first_layer = nn.Linear(
    in_features=1,
    out_features=2
)

activation=nn.ReLU()

second_layer = nn.Linear(
    in_features=2,
    out_features=1
)

with torch.no_grad():
    first_layer.weight.copy_(
        torch.tensor([
            [1.0],
            [-1.0]
        ])
    )
    first_layer.bias.zero_()

    second_layer.weight.copy_(
        torch.tensor([
            [1.0,1.0]
        ])
    )
    second_layer.bias.zero_()

hidden_values = first_layer(inputs)
outputs_without_relu = second_layer(hidden_values)

activated_values = activation(hidden_values)
outputs_with_relu = second_layer(activated_values)

print("inputs:")
print(inputs)

print("hidden values:")
print(hidden_values)

print("outputs without ReLU:")
print(outputs_without_relu)

print("values after ReLU:")
print(activated_values)

print("outputs with ReLU:")
print(outputs_with_relu)