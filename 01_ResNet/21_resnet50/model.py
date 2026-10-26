#定义独立的ResNet50v1.5模型，用于后续训练、评估与预测

import torch 
from torch import nn

#定义瓶颈残差块
class Bottleneck(nn.Module):
    expansion=4

    def __init__(self,in_channels,channels,stride=1):
        super().__init__()

        out_channels=channels*self.expansion
        self.conv1=nn.Conv2d(in_channels,channels,kernel_size=1,bias=False)
        self.bn1=nn.BatchNorm2d(channels)

        self.conv2=nn.Conv2d(channels,channels,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn2=nn.BatchNorm2d(channels)

        self.conv3=nn.Conv2d(channels,out_channels,stride=1,kernel_size=1,bias=False)
        self.bn3=nn.BatchNorm2d(out_channels)

        self.relu=nn.ReLU()

        self.downsample=None
        if stride!=1 or in_channels != out_channels:
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


#组装完整ResNet50
class ResNet50(nn.Module):
    def __init__(self,num_classes=100):
        super().__init__()

        self.conv1=nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn1=nn.BatchNorm2d(64)
        self.relu=nn.ReLU()
        self.maxpool=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        self.layer1=self._make_stage(64,64,block_count=3,stride=1)

        self.layer2=self._make_stage(256,128,block_count=4,stride=2)

        self.layer3=self._make_stage(
            512,256,block_count=6,stride=2
        )
        self.layer4=self._make_stage(
            1024,512,block_count=3,stride=2
        )

        self.avgpool=nn.AdaptiveAvgPool2d((1,1))
        self.fc=nn.Linear(2048,num_classes)

        self._initialize_weights()

    def _make_stage(self,in_channels,channels,block_count,stride):
        blocks=[Bottleneck(in_channels,channels,stride=stride)]

        out_channels=channels*Bottleneck.expansion

        for block_index in range(1,block_count):
            blocks.append(Bottleneck(out_channels,channels))

        return nn.Sequential(*blocks)

    def _initialize_weights(self):
        for module in self.modules():   #遍历大模块、小模块，ResNet50、nn.Sequential、Bottleneck、nn.Conv2d、nn.BatchNorm2d、nn.ReLU等。
            if isinstance(module,nn.Conv2d):    #判断是不是二维卷积层
                nn.init.kaiming_normal_(    #nn.init:PyTorch 提供的参数初始化工具;kaiming:何恺明等人提出的初始化方法;normal:使用正态分布生成随机数;末尾 _:直接修改传入的张量
                    module.weight,
                    mode='fan_out',     #让反向传播的梯度尺度保持稳定
                    nonlinearity='relu' #初始化时，按照适合ReLU的规则来确定随机数尺度
                )
            elif isinstance(module,nn.BatchNorm2d):
                nn.init.ones_(module.weight)    #缩放参数设为1
                nn.init.zeros_(module.bias)     #平移参数设为0

    def forward(self,images):
        output=self.conv1(images)
        output=self.bn1(output)
        output=self.relu(output)
        output=self.maxpool(output)

        output=self.layer1(output)
        output=self.layer2(output)
        output=self.layer3(output)
        output=self.layer4(output)

        output=self.avgpool(output)
        output=torch.flatten(output,1)
        output=self.fc(output)

        return output


#提供统一的模型创建入口
def resnet50(num_classes=100):
    return ResNet50(num_classes=num_classes)

            