#实验20：实现瓶颈残差块，观察三层卷积及捷径，并与参考实现核对前向。瓶颈结构让较深网络的计算更经济。

import torch 
from torch import nn 
from torchvision.models.resnet import Bottleneck as ReferenceBottleneck


#定义三层卷积瓶颈残差块
class Bottleneck(nn.Module):
    expansion=4

    def __init__(self,in_channels,channels,stride=1):
        super().__init__()

        out_channels=channels*self.expansion

        self.conv1=nn.Conv2d(in_channels,channels,kernel_size=1,bias=False)
        self.bn1=nn.BatchNorm2d(channels)

        self.conv2=nn.Conv2d(channels,channels,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn2=nn.BatchNorm2d(channels)

        self.conv3=nn.Conv2d(channels,out_channels,kernel_size=1,bias=False)
        self.bn3=nn.BatchNorm2d(out_channels)

        self.relu=nn.ReLU()

        self.downsample=None
        if stride!=1 or in_channels!=out_channels:
            self.downsample=nn.Sequential(
                nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(out_channels) 
            )

    def forward(self,block_input):
        shortcut=block_input

        output=self.conv1(block_input)
        output=self.bn1(output)
        output=self.relu(output)

        output=self.conv2(output)
        output=self.bn2(output)
        output=self.relu(output)

        output=self.conv3(output)
        output=self.bn3(output)

        if self.downsample is not None:
            shortcut=self.downsample(block_input)

        output=output+shortcut
        output=self.relu(output)

        return output

#为参考块创建对应的变换捷径
def create_reference_block(in_channels,channels,stride):
    out_channels=channels*ReferenceBottleneck.expansion

    downsample=None
    if stride != 1 or in_channels != out_channels:
        downsample = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=1, stride=stride, bias=False
            ),
            nn.BatchNorm2d(out_channels),
        )

    return ReferenceBottleneck(
        inplanes=in_channels,
        planes=channels,
        stride=stride,
        downsample=downsample
    )

#拆开一个块，观察每一个计算节点
def trace_block(block,block_input):
    print('输入：',block_input.shape)

    output=block.conv1(block_input)
    print('conv1后:',output.shape)
    output = block.bn1(output)
    output = block.relu(output)

    output = block.conv2(output)
    print("conv2后：", output.shape)
    output = block.bn2(output)
    output = block.relu(output)

    output = block.conv3(output)
    print("conv3后：", output.shape)
    output = block.bn3(output)
    print("主分支F(x)：", output.shape)

    shortcut = block_input
    if block.downsample is not None:
        shortcut = block.downsample(block_input)

    print("捷径输出：", shortcut.shape)
    print("是否使用变换捷径：", block.downsample is not None)

    output = output + shortcut
    output = block.relu(output)

    print("块输出：",output.shape)
    return output

#验证恒等捷径、只改通道、同时改变通道和空间尺寸
torch.manual_seed(42)
device=torch.device('cuda'if torch.cuda.is_available()else 'cpu')

cases=[
    ('恒等捷径',256,64,1),
    ('只改变通道',64,64,1),
    ('改变通道和空间尺寸',256,128,2)
]

print('计算设备',device)

for case_name,in_channels,channels,stride in cases:
    print(f"\n情况:{case_name}")

    block=Bottleneck(
        in_channels,channels,stride
    ).to(device)

    reference_block = create_reference_block(
        in_channels, channels, stride
    ).to(device)

    load_result = reference_block.load_state_dict(block.state_dict())

    block.eval()
    reference_block.eval()

    block_input = torch.randn(
        2, in_channels, 8, 8, device=device
    )

    with torch.no_grad():
        manual_output = trace_block(block, block_input)
        direct_output = block(block_input)
        reference_output = reference_block(block_input)

    print("状态加载结果：", load_result)
    print(
        "手动展开与完整调用是否一致：",
        torch.allclose(manual_output, direct_output),
    )
    print(
        "自定义块与参考块是否一致：",
        torch.allclose(direct_output, reference_output),
    )
    print(
        "与参考块的最大绝对差：",
        (direct_output - reference_output).abs().max().item(),
    )

