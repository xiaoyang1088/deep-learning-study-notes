#本文件使用SGD优化器重复更新权重和偏置
#实验观察预测值逐渐接近目标值、损失逐渐下降过程

import torch
from torch import nn

input_value = torch.tensor(3.0)
target = torch.tensor(10.0)

weight = nn.Parameter(torch.tensor(2.0))
bias = nn.Parameter(torch.tensor(1.0))

optimizer = torch.optim.SGD(
    [weight,bias],
    lr=0.01
)

for epoch in range(1,11):
    optimizer.zero_grad()

    prediction=input_value*weight+bias
    loss=(prediction-target)**2

    loss.backward()

    print(
        f"epoch {epoch:2d} | "
        f"prediction={prediction.item():.4f} | "
        f"loss={loss.item():.4f} | "
        f"weight={weight.item():.4f} | "
        f"bias={bias.item():.4f}"
    )

    optimizer.step()

with torch.no_grad():
    final_prediction=input_value*weight+bias
    final_loss=(prediction-target)**2

print("final prediction:", final_prediction.item())
print("final loss:", final_loss.item())
print("final weight:", weight.item())
print("final bias:", bias.item())