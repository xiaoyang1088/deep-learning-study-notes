#本文件使用一个简单函数演示pytorch自动求导
#实验观察损失对权重和偏置的梯度

import torch

input_value = torch.tensor(3.0)
target = torch.tensor(10.0)

weight = torch.tensor(
    2.0,
    requires_grad=True
)

bias = torch.tensor(
    1.0,
    requires_grad=True
)

prediction = input_value * weight + bias
loss=(prediction - target) ** 2

loss.backward()

print("input:", input_value)
print("target:", target)
print("weight:", weight)
print("bias:", bias)
print("prediction:", prediction)
print("loss:", loss)
print("weight gradient:", weight.grad)
print("bias gradient:", bias.grad)