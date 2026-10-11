# 实验 08：用已经实现的基本块，组装完整 ResNet18，并与 torchvision 的实现核对

#组合基本残差块构建ResNet18，追踪各阶段并与参考实现核对
import torch
from torch import nn 
from torchvision.models import resnet18



#定义基本残差块
class BasicBlock(nn.Module):
    def __init__(self, in_channels,out_channels,stride=1):
        super().__init__() 

        self.conv1=nn.Conv2d(
            in_channels,out_channels,kernel_size=3,
            stride=stride,padding=1,bias=False
        )
        self.bn1=nn.BatchNorm2d(out_channels)
        self.relu=nn.ReLU()

        self.conv2=nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn2=nn.BatchNorm2d(out_channels)

        self.downsample=None
        if stride !=1 or in_channels!=out_channels:
            self.downsample=nn.Sequential(
                nn.Conv2d(
                    in_channels,out_channels,kernel_size=1,stride=stride,bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )


    def forward(self,block_input):
        shortcut=block_input

        output=self.conv1(block_input)
        output=self.bn1(output)
        output=self.relu(output)

        output=self.conv2(output)
        output=self.bn2(output)

        if self.downsample is not None:
            shortcut=self.downsample(block_input)

        output=output+shortcut
        output=self.relu(output)

        return output 

#组合入口，四个残差阶段和分类
class ResNet18(nn.Module):
    def __init__(self,num_classes=1000):
        super().__init__()

        self.conv1=nn.Conv2d(
            3,64,kernel_size=7,stride=2,padding=3,bias=False
        )
        self.bn1=nn.BatchNorm2d(64)
        self.relu=nn.ReLU()
        self.maxpool=nn.MaxPool2d(
            kernel_size=3,stride=2,padding=1
        )

        self.layer1=nn.Sequential(
            BasicBlock(64,64),
            BasicBlock(64,64)
        )

        self.layer2=nn.Sequential(
            BasicBlock(64,128,stride=2),
            BasicBlock(128,128)
        )

        self.layer3=nn.Sequential(
            BasicBlock(128,256,stride=2),
            BasicBlock(256,256)
        )

        self.layer4=nn.Sequential(
            BasicBlock(256,512,stride=2),
            BasicBlock(512,512)
        )

        self.avgpool=nn.AdaptiveAvgPool2d((1,1))
        self.fc=nn.Linear(512,num_classes)

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

#创建两个模型，并对齐参数与BN运行统计量
model=ResNet18(num_classes=1000)
reference_model=resnet18(weights=None)

load_result=reference_model.load_state_dict(model.state_dict())
print("状态加载结果：",load_result)

model.eval()
reference_model.eval()

images=torch.randn(2,3,224,224)

with torch.no_grad():
    output=model(images)
    reference_output=reference_model(images)

print("输入：", images.shape)
print("自定义模型与参考模型是否一致：",
      torch.allclose(output, reference_output))
print("与参考模型的最大绝对差：",
      (output - reference_output).abs().max().item())