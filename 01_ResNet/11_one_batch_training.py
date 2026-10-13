# 实验11：用:ResNet18处理一个批次，计算损失、清除旧梯度、反向传播，更新一次参数
# 继续复用实验08中自己实现的ResNet18

from pathlib import Path
import runpy

import torch
from torch import nn

#复用实验08的模型定义
model_path=Path(__file__).resolve().parent/"08_build_resnet18.py"
experiment=runpy.run_path(str(model_path))
ResNet18=experiment['ResNet18']

#创建适合小图片的十分类模型

torch.manual_seed(42)
device=torch.device("cuda"if torch.cuda.is_available()else"cpu")

model=ResNet18(num_classes=10)
model.conv1=nn.Conv2d(
    3,64,kernel_size=3,stride=1,padding=1,bias=False
)
model.maxpool=nn.Identity()
model=model.to(device)
model.train()

#准备一个模拟批次
images=torch.rand(4,3,32,32,device=device)
labels=torch.randint(0,10,(4,),device=device)

#创建损失函数与优化器，保存更新前的分类层权重
criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(model.parameters(),lr=0.01)
weight_before=model.fc.weight.detach().clone()

#清除旧梯度，完成前向计算和损失计算
optimizer.zero_grad(set_to_none=True)
logits = model(images)
loss=criterion(logits,labels)

print("计算设备：", device)
print("图片形状：", images.shape)
print("标签：", labels)
print("分类分数形状：", logits.shape)
print("本次损失：", loss.item())
print("反向传播前分类层权重的梯度：", model.fc.weight.grad)

#反向传播，检查梯度和权重
loss.backward()

print("分类层权重梯度形状：", model.fc.weight.grad.shape)
print("分类层权重梯度平均绝对值：",
      model.fc.weight.grad.abs().mean().item())
print("反向传播后权重是否仍未改变：",
      torch.equal(weight_before, model.fc.weight.detach()))

#更新一次参数，检查实际变化
optimizer.step()

weight_after = model.fc.weight.detach()
print("更新后权重是否改变：",
      not torch.equal(weight_before, weight_after))
print("分类层权重最大绝对变化：",
      (weight_after - weight_before).abs().max().item())