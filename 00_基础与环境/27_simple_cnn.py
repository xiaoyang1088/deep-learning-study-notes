#将卷积、ReLU、池化、展平和线性层组合成一个最小CNN
#使用模拟图像观察完整前向传播及各层形状，本实验不进行训练

import torch 
from torch import nn 


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv=nn.Conv2d(
            in_channels=1,
            out_channels=2,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.activation=nn.ReLU()

        self.pool=nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        self.flatten=nn.Flatten(start_dim=1)

        self.classifier=nn.Linear(
            in_features=8,
            out_features=3
        )

    def forward(self,inputs):
        print("输入形状：", inputs.shape)

        features = self.conv(inputs)
        print("卷积后形状：", features.shape)

        features = self.activation(features)
        print("ReLU 后形状：", features.shape)

        features = self.pool(features)
        print("池化后形状：", features.shape)

        features = self.flatten(features)
        print("展平后形状：", features.shape)

        logits = self.classifier(features)
        print("分类分数形状：", logits.shape)

        return logits

torch.manual_seed(0)

image = torch.tensor([
    [1.0, 2.0, 3.0, 4.0],
    [5.0, 6.0, 7.0, 8.0],
    [9.0, 10.0, 11.0, 12.0],
    [13.0, 14.0, 15.0, 16.0],
], dtype=torch.float32)

inputs = image.reshape(1, 1, 4, 4)

model = SimpleCNN()
model.eval()

print("模型结构：")
print(model)

print("\n前向传播：")
with torch.no_grad():
    logits = model(inputs)
    probabilities = torch.softmax(logits, dim=1)
    predicted_classes = torch.argmax(logits, dim=1)

print("\n原始分类分数：")
print(logits)

print("\n类别概率：")
print(probabilities)

print("\n预测类别编号：")
print(predicted_classes)

print("\n可训练参数的名称和形状：")
for name, parameter in model.named_parameters():
    print(name, parameter.shape)