#学习交叉熵损失如何使用分类分数和真实类别标签
#对照手工计算，并观察提高正确类别分数后损失的变化

import torch
from torch import nn

logits=torch.tensor([
    [2.0,1.0,0.0],
    [0.0,1.0,3.0]
])

targets=torch.tensor(
    [0,2],
    dtype=torch.long
)

loss_function = nn.CrossEntropyLoss()
loss=loss_function(logits,targets)

probabilities=torch.softmax(logits,dim=1)

correct_probabilities=torch.stack([
    probabilities[0,0],
    probabilities[1,2]
])

individual_losses=-torch.log(correct_probabilities)

manual_loss=individual_losses.mean()

improved_logits=torch.tensor([
    [4.0,1.0,0.0],
    [0.0,1.0,5.0]
])

improved_loss=loss_function(improved_logits,targets)


print("原始分数：")
print(logits)
print("分数形状：", logits.shape)

print("\n真实类别标签：")
print(targets)
print("标签形状：", targets.shape)
print("标签数据类型：", targets.dtype)

print("\n类别概率：")
print(probabilities)

print("\n每个样本的正确类别概率：")
print(correct_probabilities)

print("\n每个样本的损失：")
print(individual_losses)

print("\nPyTorch 计算的平均损失：", loss.item())
print("手工计算的平均损失：", manual_loss.item())
print("提高正确类别分数后的平均损失：", improved_loss.item())