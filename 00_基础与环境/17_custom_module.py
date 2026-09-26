#本文件使用nn.Moudle把两个线性层和ReLU封装成完整模型
#模型沿用上一个实验的固定参数，用于计算输入数字的绝对值

import torch
from torch import nn

class AbsoluteValueNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.first_layer=nn.Linear(
            in_features=1,
            out_features=2
        )

        self.activation=nn.ReLU()

        self.second_layer=nn.Linear(
            in_features=2,
            out_features=1
        )

        with torch.no_grad():
            self.first_layer.weight.copy_(
                torch.tensor([
                    [1.0],
                    [-1.0],
                ])
            )
            self.first_layer.bias.zero_()

            self.second_layer.weight.copy_(
                torch.tensor([
                    [1.0, 1.0],
                ])
            )
            self.second_layer.bias.zero_()

    def forward(self,inputs):
        hidden_values=self.first_layer(inputs)
        activated_values=self.activation(hidden_values)
        outputs = self.second_layer(activated_values)

        return outputs


inputs=torch.tensor([
    [-2.0],
    [-1.0],
    [0.0],
    [1.0],
    [2.0]
])

model = AbsoluteValueNetwork()
outputs = model(inputs)

print("model structure:")
print(model)

print("inputs:")
print(inputs)

print("outputs:")
print(outputs)

print("named parameters:")
for name,parameter in model.named_parameters():
    print(name,parameter.shape)