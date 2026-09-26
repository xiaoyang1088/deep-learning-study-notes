#本文件根据自动求导得到的梯度，手工更新一次权重和偏置
#实验比较参数更新前后的预测值和损失

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

learning_rate=0.01

initial_prediction=input_value*weight+bias
initial_loss=(initial_prediction-target)**2

initial_loss.backward()

print("before update:")
print("weight:",weight)
print("bias:", bias)
print("prediction:", initial_prediction)
print("loss:", initial_loss)
print("weight gradient:", weight.grad)
print("bias gradient:", bias.grad)

with torch.no_grad():
    weight-=learning_rate*weight.grad
    bias-=learning_rate*bias.grad

weight.grad.zero_()
bias.grad.zero_()

new_prediction=input_value*weight+bias
new_loss = (new_prediction-target)**2

print("after update:")
print("weight:", weight)
print("bias:", bias)
print("prediction:", new_prediction)
print("loss:", new_loss)
print("weight gradient:", weight.grad)
print("bias gradient:", bias.grad)