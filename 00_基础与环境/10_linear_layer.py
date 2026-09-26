#本文件将nn.Linear的输出与手工矩阵乘法结果进行比较

import torch
from torch import nn

torch.manual_seed(0)

features=torch.tensor([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0]
])

linear_layer=nn.Linear(
    in_features=3,
    out_features=4,
    bias=True
)

layer_outputs=linear_layer(features)

manual_outputs=(
    features@linear_layer.weight.T+linear_layer.bias
)

print("features shape:", features.shape)
print("weight shape:", linear_layer.weight.shape)
print("bias shape:", linear_layer.bias.shape)
print("outputs shape:", layer_outputs.shape)

print("layer outputs:")
print(layer_outputs)

print("manual outputs:")
print(manual_outputs)

print(
    "results are equal:",
    torch.allclose(layer_outputs, manual_outputs),
)