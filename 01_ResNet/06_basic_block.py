# 实验 05 中手动指定的F(x)，换成真正经过两次卷积和 BN 计算的结果，再将整个过程封装成一个能调用的残差块

#自定义不改变尺寸的基本残差块，逐层追踪并核对完整调用的输出
import torch
from torch import nn


#定义基本残差块

class BasicBlock(nn.Module):
    def __init__(self,channels):
        super().__init__()

        self.conv1=nn.Conv2d(
            channels,channels,kernel_size=3,stride=1,padding=1,bias=False
        )

        self.bn1=nn.BatchNorm2d(channels)
        self.relu=nn.ReLU()

        self.conv2=nn.Conv2d(
            channels, channels, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn2=nn.BatchNorm2d(channels)

    def forward(self,block_input):
        shortcut=block_input

        output=self.conv1(block_input)
        output=self.bn1(output)
        output=self.relu(output)

        output=self.conv2(output)
        output=self.bn2(output)

        output=output+shortcut
        output=self.relu(output)

        return output

#创建残差块和模拟输入
block=BasicBlock(channels=4)
block.eval()

block_input = torch.randn(2, 4, 5, 5)

# 拆开残差块，逐层观察输出
with torch.no_grad():
    conv1_output = block.conv1(block_input)
    bn1_output = block.bn1(conv1_output)
    relu1_output = block.relu(bn1_output)

    conv2_output = block.conv2(relu1_output)
    residual_output = block.bn2(conv2_output)

    shortcut_output = block_input
    added_output = residual_output + shortcut_output
    manual_output = block.relu(added_output)

    direct_output = block(block_input)

print("残差块结构：", block)
print("输入形状：", block_input.shape)
print("第一次卷积后：", conv1_output.shape)
print("第一次 BN 后：", bn1_output.shape)
print("第一次 ReLU 后：", relu1_output.shape)
print("第二次卷积后：", conv2_output.shape)
print("第二次 BN 后，即残差分支输出：", residual_output.shape)
print("捷径输出：", shortcut_output.shape)
print("相加后：", added_output.shape)
print("末尾 ReLU 后：", manual_output.shape)

# 核对逐层计算与完整调用
print("两路形状是否一致：",
      residual_output.shape == shortcut_output.shape)
print("逐层计算与完整调用是否一致：",
      torch.allclose(manual_output, direct_output))
print("最大绝对差：",
      (manual_output - direct_output).abs().max().item())