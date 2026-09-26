#本文件使用均方误差衡量线性层预测值和目标值之间的差距

import torch
from torch import nn

torch.manual_seed(0)

features=torch.tensor([
    [1.0,2.0,3.0],
    [4.0,5.0,6.0]
])

targets=torch.tensor([
    [1.0],
    [0.0]
])

model= nn.Linear(
    in_features=3,
    out_features=1
)

predictions = model(features)

loss_function=nn.MSELoss()
loss=loss_function(predictions,targets)

manual_loss=((predictions-targets)**2).mean()

print("features shape:", features.shape)
print("targets shape:", targets.shape)
print("predictions shape:", predictions.shape)

print("predictions:")
print(predictions)

print("targets:")
print(targets)

print("loss:", loss)
print("manual loss:", manual_loss)
print("results are equal:", torch.allclose(loss, manual_loss))
