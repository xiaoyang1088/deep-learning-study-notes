#使用nn.sequential按顺序组织两个线性层和ReLU
#手动设置参数计算绝对值，并观察各层输出和张量形状

import torch
from torch import nn 


inputs=torch.tensor([
    [-2.0],
    [-1.0],
    [0.0],
    [1.0],
    [2.0]
])

model = nn.Sequential(
    nn.Linear(in_features=1,out_features=2),
    nn.ReLU(),
    nn.Linear(in_features=2,out_features=1)
)

with torch.no_grad():
    model[0].weight.copy_(
        torch.tensor([
            [1.0],[-1.0]
        ])
    )
    model[0].bias.zero_()

    model[2].weight.copy_(
        torch.tensor([
            [1.0, 1.0],
        ])
    )
    model[2].bias.zero_()

outputs = model(inputs)

print("模型结构：")
print(model)

print("\n输入：")
print(inputs)
print("输入形状：", inputs.shape)

print("\n输出：")
print(outputs)
print("输出形状：", outputs.shape)

print("\n逐层观察数据：")
current_values=inputs

for layer_index , layer in enumerate(model):
    current_values=layer(current_values)
    print("模块编号：",layer_index)
    print("模块结构：",layer)
    print("输出数据：",current_values)
    print("输出形状：",current_values.shape)

print("\n模型参数：")
for name ,parameter in model.named_parameters():
    print(name,parameter.shape)