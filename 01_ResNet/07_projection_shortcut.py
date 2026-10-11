#实验07：投影捷径实验

#实现自动选择捷径的基本残差块，验证不同通道数和步幅下的输出

import torch
from torch import nn 


#定义支持形状变化的基本残差块

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # downsample是下采样，也就是减少空间采样位置
        self.downsample=None

        if stride !=1 or in_channels != out_channels:
            #sequential意为按顺序的
            self.downsample=nn.Sequential(
                nn.Conv2d(
                    in_channels,out_channels,kernel_size=1,stride=stride,bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self,block_input):
        shortcut = block_input

        output = self.conv1(block_input)
        output = self.bn1(output)
        output = self.relu(output)

        output = self.conv2(output)
        output = self.bn2(output)

        if self.downsample is not None:
            shortcut = self.downsample(block_input)

        output = output + shortcut
        output = self.relu(output)

        return output


#分别验证恒等捷径，只改变通道，同时改变通道和空间尺寸

test_cases=[
    ("恒等捷径",4,4,1),
    ("只改变通道",4,8,1),
    ("同时改变通道和空间尺寸",4,8,2)
]

for case_name, in_channels,out_channels,stride in test_cases:
    block=BasicBlock(in_channels,out_channels,stride)
    block.eval()

    #torch.rand在 [0,1) 内均匀取值，不出现负值
    #torch.randn按标准正态分布取值，集中在 0 附近，能出现负值
    block_input=torch.randn(2,in_channels,6,6)

    #拆分主分支和捷径，核对完整调用
    with torch.no_grad():
        conv1_output = block.conv1(block_input)
        bn1_output = block.bn1(conv1_output)
        relu1_output = block.relu(bn1_output)

        conv2_output = block.conv2(relu1_output)
        residual_output = block.bn2(conv2_output)

        shortcut_output = block_input
        if block.downsample is not None:
            shortcut_output = block.downsample(block_input)

        added_output = residual_output + shortcut_output
        manual_output = block.relu(added_output)

        direct_output = block(block_input)

    print("\n实验情况：", case_name)
    print("捷径模块：", block.downsample)
    print("输入形状：", block_input.shape)
    print("第一次卷积后：", conv1_output.shape)
    print("第二次卷积后：", conv2_output.shape)
    print("残差分支输出：", residual_output.shape)
    print("捷径输出：", shortcut_output.shape)
    print("完整块输出：", direct_output.shape)
    print("两路形状是否一致：",
          residual_output.shape == shortcut_output.shape)
    print("逐层计算与完整调用是否一致：",
          torch.allclose(manual_output, direct_output))
    print("最大绝对差：",
          (manual_output - direct_output).abs().max().item())