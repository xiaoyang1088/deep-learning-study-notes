#实验19，复用基本残差块构建ResNet34，比较阶段结构并与参考实现核对前向

from pathlib import Path
import runpy

import torch
from torch import nn
from torchvision.models import resnet34

#复用已经实现的基本块和ResNet18
base_dir=Path(__file__).resolve().parent
experiment=runpy.run_path(str(base_dir/'08_build_resnet18.py'))

BasicBlock=experiment['BasicBlock']
ResNet18=experiment['ResNet18']

#按指定数量组装一个残差阶段
def build_stage(in_channels,out_channels,block_count,stride):
    blocks=[
        BasicBlock(in_channels,out_channels,stride)
    ]       

    for block_index in range(1,block_count):
        blocks.append(BasicBlock(out_channels,out_channels))

    return nn.Sequential(*blocks)   #*解列表

#组装标准输入版的ResNet34
class ResNet34(nn.Module):
    def __init__(self,num_classes=1000):
        super().__init__()

        self.conv1=nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn1=nn.BatchNorm2d(64)
        self.relu=nn.ReLU()
        self.maxpool=nn.MaxPool2d(
            kernel_size=3,stride=2,padding=1
        )

        self.layer1=build_stage(64,64,block_count=3,stride=1)
        self.layer2=build_stage(64,128,block_count=4,stride=2)
        self.layer3=build_stage(128,256,block_count=6,stride=2)
        self.layer4=build_stage(256,512,block_count=3,stride=2)

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


#创建模型并对齐ResNet34参考实现的状态
torch.manual_seed(42)
device=torch.device('cuda'if torch.cuda.is_available()else 'cpu')
model18=ResNet18(num_classes=1000).to(device)
model34=ResNet34(num_classes=1000).to(device)
reference_model=resnet34(weights=None).to(device)

load_result=reference_model.load_state_dict(model34.state_dict())

model18.eval()
model34.eval()
reference_model.eval()

print('计算设备：',device)
print('状态加载结果',load_result)

#比较各阶段块数、命名层数和参数量
for model_name,model in [('ResNet18',model18),('ResNet34',model34)]:
    block_counts=[
        len(model.layer1),
        len(model.layer2),
        len(model.layer3),
        len(model.layer4)
    ]
    named_depth=1+2*sum(block_counts)+1
    parameter_count=sum(parameter.numel()for parameter in model.parameters())   #numel:number of elements,计算这个张量中有多少个数

    print(f"{model_name}:阶段块数={block_counts},命名层数={named_depth},参数量={parameter_count:,}")    #:,表示显示数字时，每三位加一个逗号


#追踪ResNet34个阶段前向形状
images=torch.randn(2,3,224,224,device=device)       #randn 生成的是标准正态分布的随机数

with torch.no_grad():
    print('输入',images.shape)

    output = model34.conv1(images)
    output = model34.bn1(output)
    output = model34.relu(output)
    output = model34.maxpool(output)
    print("入口处理后：", output.shape)

    output = model34.layer1(output)
    print("layer1：", output.shape)

    output = model34.layer2(output)
    print("layer2：", output.shape)

    output = model34.layer3(output)
    print("layer3：", output.shape)

    output = model34.layer4(output)
    print("layer4：", output.shape)

    output = model34.avgpool(output)
    print("平均池化后：", output.shape)

    output = torch.flatten(output, 1)
    print("展平后：", output.shape)

    manual_output = model34.fc(output)
    print("分类分数：", manual_output.shape)

    direct_output = model34(images)
    reference_output = reference_model(images)
    output18 = model18(images)   

# 核对手动展开与完整模型、参考模型的计算
print(
    "手动展开与完整调用是否一致：",
    torch.allclose(manual_output, direct_output),
)
print(
    "自定义模型与参考模型是否一致：",
    torch.allclose(direct_output, reference_output),
)
print(
    "与参考模型的最大绝对差：",
    (direct_output - reference_output).abs().max().item(),
)

print("ResNet18输出形状：", output18.shape)
print("ResNet34输出形状：", direct_output.shape)