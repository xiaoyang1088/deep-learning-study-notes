#使用Fashion-MNIST训练一个小型卷积神经网络
#从官方训练集中划分训练集和验证集，观察每轮损失与验证准确率
#本实验不使用官方测试集，训练结束后保存最后一轮的模型参数

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader,random_split
from torchvision import datasets,transforms

class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv=nn.Conv2d(
            in_channels=1,
            out_channels=8,kernel_size=3,
            stride=1,
            padding=1
        )

        self.activation=nn.ReLU()
        self.pool=nn.MaxPool2d(kernel_size=2,stride=2)
        self.flatten=nn.Flatten(start_dim=1)
        self.classifier=nn.Linear(
            in_features=8*14*14,
            out_features=10
        )

    def forward(self,inputs):
        features=self.conv(inputs)
        features=self.activation(features)
        features=self.pool(features)
        features=self.flatten(features)
        logits=self.classifier(features)
        return logits


torch.manual_seed(0)

device = torch.device("cuda"if torch.cuda.is_available()else "cpu")

experiment_dir=Path(__file__).resolve().parent
data_dir=experiment_dir/"data"

full_dataset=datasets.FashionMNIST(
    root=str(data_dir),
    train=True,
    transform=transforms.ToTensor(),
    download=False
)

split_genetator=torch.Generator().manual_seed(0)

train_dataset,validation_dataset=random_split(
    full_dataset,
    [50000,10000],
    generator=split_genetator
)

train_loader=DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=0
)

validation_loader=DataLoader(
    validation_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=0
)

model=ImageClassifier().to(device)
loss_fuction=nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(model.parameters(),lr=0.1)

print("计算设备：", device)
print("训练集样本数：", len(train_dataset))
print("验证集样本数：", len(validation_dataset))
print("每轮训练批次数：", len(train_loader))
print("模型结构：")
print(model)

for epoch in range(3):
    model.train()
    training_loss_sum=0.0

    for batch_images,batch_labels in train_loader:
        batch_images=batch_images.to(device)
        batch_labels=batch_labels.to(device)

        optimizer.zero_grad()

        logits=model(batch_images)
        loss=loss_fuction(logits,batch_labels)

        loss.backward()
        optimizer.step()

        training_loss_sum+=loss.item()*batch_labels.shape[0]

    average_training_loss=training_loss_sum/len(train_dataset)

    model.eval()
    validation_loss_sum = 0.0
    correct_count=0

    with torch.no_grad():
        for batch_images,batch_labels in validation_loader:
            batch_images=batch_images.to(device)
            batch_labels=batch_labels.to(device)

            logits = model(batch_images)
            loss=loss_fuction(logits,batch_labels)

            validation_loss_sum+=loss.item()*batch_labels.shape[0]

            predicitons=torch.argmax(logits,dim=1)
            correct_count+=(predicitons==batch_labels).sum().item()

    average_validation_loss=validation_loss_sum/len(validation_dataset)
    validation_accuracy=correct_count/len(validation_dataset)

    print(
        f"轮次：{epoch + 1} "
        f"训练损失：{average_training_loss:.4f} "
        f"验证损失：{average_validation_loss:.4f} "
        f"验证准确率：{validation_accuracy:.2%}"
    )

checkpoint_dir=experiment_dir/"checkpoints"
checkpoint_dir.mkdir(parents=True,exist_ok=True)

weight_path=checkpoint_dir/"fashion_mnist_cnn_last.pth"

torch.save(model.state_dict(),weight_path)

print("模型参数已保存到：",weight_path)   