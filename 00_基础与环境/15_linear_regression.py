#本文件训练一个最小线性回归模型
#模型通过多组输入和目标，学习y=2x+1的关系

import torch
from torch import nn

torch.manual_seed(0)

inputs=torch.tensor([
    [0.0],
    [1.0],
    [2.0],
    [3.0],
    [4.0]
])

targets=2*inputs+1

model=nn.Linear(
    in_features=1,
    out_features=1
)

loss_function=nn.MSELoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.05
)

for epoch in range(1,201):
    optimizer.zero_grad()

    predictions=model(inputs)
    loss = loss_function(predictions,targets)

    loss.backward()
    optimizer.step()

    if epoch == 1 or epoch%20==0:
        print(
            f"epoch={epoch:3d} | "
            f"loss={loss.item():.6f} | "
            f"weight={model.weight.item():.4f} | "
            f"bias={model.bias.item():.4f}"
        )

with torch.no_grad():
    test_input = torch.tensor([[5.0]])
    test_prediction = model(test_input)

print("learned weight:", model.weight.item())
print("learned bias:", model.bias.item())
print("prediction for x=5:", test_prediction.item())