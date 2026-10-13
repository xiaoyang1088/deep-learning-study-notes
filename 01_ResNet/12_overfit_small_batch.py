# 实验12：让模型学会一小组真实图片,充分拟合一个小批次

# 固定32张真实训练图片，反复训练并观察损失与准确率

from pathlib import Path
import runpy

import torch 
from torch import nn
from torch.utils.data import DataLoader,Subset,random_split
from torchvision import datasets,transforms

#复用实验08的自定义模型，设置设备与随机种子
model_path=Path(__file__).resolve().parent/"08_build_resnet18.py"
experiment=runpy.run_path(str(model_path))
ResNet18=experiment['ResNet18']

torch.manual_seed(42)
device=torch.device("cuda"if torch.cuda.is_available()else "cpu")

#沿用实验10的训练和验证划分，取出固定32张训练图片
data_dir=Path(__file__).resolve().parent/"data"

full_train_dataset=datasets.CIFAR10(
    root=str(data_dir),
    train=True,
    transform=transforms.ToTensor(),
    download=False
)

train_dataset,val_dataset=random_split(
    full_train_dataset,
    [45000,5000],
    generator=torch.Generator().manual_seed(42)
)

sample_count=32
small_dataset=Subset(train_dataset,range(sample_count))
small_loader=DataLoader(
    small_dataset,
    batch_size=sample_count,
    shuffle=False,
    num_workers=0
)

images,labels=next(iter(small_loader))
images=images.to(device)
labels=labels.to(device)

#创建小图片版ResNet18、损失函数和优化器
model=ResNet18(num_classes=10)
model.conv1=nn.Conv2d(
    3,64,kernel_size=3,stride=1,padding=1,bias=False
)
model.maxpool = nn.Identity()
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

#在评估模式下检查这组固定图片
def evaluate_fixed_batch(model,images,labels,criterion):
    model.eval()
    with torch.no_grad():
        logits=model(images)
        loss=criterion(logits,labels)
        predictions=logits.argmax(dim=1)
        accuracy=(predictions==labels).float().mean().item()
    return loss.item(),accuracy

print("计算设备：",device)
print("固定图片形状：",images.shape)
print("固定标签：",labels)

initial_loss,initial_accuracy=evaluate_fixed_batch(
    model,images,labels,criterion
)

print(
    f"训练前：评估模式损失={initial_loss:.4f},固定样本准确率={initial_accuracy:.2%}"
)

#反复使用同一组图片更新参数

training_steps=100
for step in range(1,training_steps+1):
    model.train()
    optimizer.zero_grad(0)

    logits=model(images)
    loss=criterion(logits,labels)

    loss.backward()
    optimizer.step()

    if step==1 or step%10==0:
        eval_loss,eval_accuracy=evaluate_fixed_batch(model,images,labels,criterion)
        print(
            f'更新次数={step:3d},更新前训练模式损失={loss.item():.4f},更新后评估模式损失={eval_loss:.4f},固定样本准确率={eval_accuracy:.2%}'
        )