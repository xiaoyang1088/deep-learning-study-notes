#实验04，批量归一化
# 用一组很小、能手算的数据观察：训练模式下，BN 如何根据当前输入算出均值、方差，再调整数值

#用可手算的小张量观察BatchNorm2d在训练模式下的归一化结果
import torch
from torch import nn

features=torch.tensor(
    [
        [[[1.0,3.0]]],
        [[[5.0,7.0]]]
    ]
)

bn=nn.BatchNorm2d(num_features=1)     #num_features=1 在这里表示输入有一个通道
bn.train()

with torch.no_grad():
    train_output=bn(features)

print("输入形状：", features.shape)
print("输入数值：", features)
print("BN 是否处于训练模式：", bn.training)
print("输出形状：", train_output.shape)
print("训练模式输出：", train_output)

#查看一次训练模式前向计算后保存的统计量
print("\n训练前向后的运行均值：",bn.running_mean)
print("训练前向后的运行方差：",bn.running_var)

#切换到评估模式，用保存的统计量处理同一批输入
bn.eval()

with torch.no_grad():
    eval_output=bn(features)


print("\nBN 是否处于训练模式：", bn.training)
print("之前的训练模式输出：", train_output)
print("现在的评估模式输出：", eval_output)

print("评估前向后的运行均值：", bn.running_mean)
print("评估前向后的运行方差：", bn.running_var)

#BN的最后两步：乘一个可学习的数，再加一个可学习的数,本次手动设置

#查看BN的可学习缩放参数和偏移参数

print("\n修改前的缩放参数gamma:",bn.weight)
print("修改前的偏移参数beta:",bn.bias)

#保持评估模式，手动设置gamma=2、beta=1
with torch.no_grad():
    bn.weight.fill_(2.0)
    bn.bias.fill_(1.0)
    adjusted_output=bn(features)

print("修改后的缩放参数 gamma：", bn.weight)
print("修改后的偏移参数 beta：", bn.bias)

print("\n修改前的评估输出：", eval_output)
print("修改后的评估输出：", adjusted_output)
print("按原输出乘2再加1：", eval_output * 2 + 1)

print(
    "参数调整后的结果是否符合公式：",
    torch.allclose(adjusted_output, eval_output * 2 + 1),
)


#用两个通道验证BN按通道独立统计和归一化

two_channel_features=torch.tensor(
    [
        [
            [[1.0, 3.0]],
            [[101.0, 103.0]],
        ],
        [
            [[5.0, 7.0]],
            [[105.0, 107.0]],
        ]
    ]
)

two_channel_bn=nn.BatchNorm2d(num_features=2)
two_channel_bn.train()

with torch.no_grad():
    two_channel_output = two_channel_bn(two_channel_features)

print("\n双通道输入形状：", two_channel_features.shape)
print("双通道输出形状：", two_channel_output.shape)

print("通道0的输入：", two_channel_features[:, 0])
print("通道1的输入：", two_channel_features[:, 1])
print("通道0的输出：", two_channel_output[:, 0])
print("通道1的输出：", two_channel_output[:, 1])

print("每个通道的缩放参数：", two_channel_bn.weight)
print("每个通道的偏移参数：", two_channel_bn.bias)