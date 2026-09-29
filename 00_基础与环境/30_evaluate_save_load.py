#加载实验29保存的模型参数，在Fashion-MNIST官方测试集评估
#再对一张测试图片进行预测，查看真实类别、预测类别与类别概率


from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=8,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.activation = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten(start_dim=1)
        self.classifier = nn.Linear(
            in_features=8 * 14 * 14,
            out_features=10,
        )

    def forward(self, inputs):
        features = self.conv(inputs)
        features = self.activation(features)
        features = self.pool(features)
        features = self.flatten(features)
        logits = self.classifier(features)
        return logits


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

experiment_dir = Path(__file__).resolve().parent
data_dir = experiment_dir / "data"
weight_path = experiment_dir / "checkpoints" / "fashion_mnist_cnn_last.pth"

model = ImageClassifier()

state_dict=torch.load(
    weight_path,
    map_location="cpu",
    weights_only=True
)

model.load_state_dict(state_dict)
model = model.to(device)
model.eval()

print("计算设备：", device)
print("已加载权重：", weight_path)

test_dataset = datasets.FashionMNIST(
    root=str(data_dir),
    train=False,
    transform=transforms.ToTensor(),
    download=False,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=0,
)

loss_function = nn.CrossEntropyLoss()
test_loss_sum = 0.0
correct_count = 0


with torch.no_grad():
    for batch_images, batch_labels in test_loader:
        batch_images = batch_images.to(device)
        batch_labels = batch_labels.to(device)

        logits = model(batch_images)
        loss = loss_function(logits, batch_labels)

        test_loss_sum += loss.item() * batch_labels.shape[0]

        predictions = torch.argmax(logits, dim=1)
        correct_count += (predictions == batch_labels).sum().item()

average_test_loss = test_loss_sum / len(test_dataset)
test_accuracy = correct_count / len(test_dataset)

print("测试集样本数：", len(test_dataset))
print("预测正确数量：", correct_count)
print(f"测试损失：{average_test_loss:.4f}")
print(f"测试准确率：{test_accuracy:.2%}")

class_names = [
    "T恤或上衣",
    "裤子",
    "套头衫",
    "连衣裙",
    "外套",
    "凉鞋",
    "衬衫",
    "运动鞋",
    "包",
    "短靴",
]

sample_index = 0
image, label = test_dataset[sample_index]
single_image = image.unsqueeze(0).to(device)

with torch.no_grad():
    single_logits = model(single_image)
    probabilities = torch.softmax(single_logits, dim=1)
    predicted_class = torch.argmax(single_logits, dim=1).item()

predicted_probability = probabilities[0, predicted_class].item()

print("\n单张图片预测：")
print("样本编号：", sample_index)
print("原始图片形状：", image.shape)
print("送入模型的形状：", single_image.shape)
print("真实类别：", label, class_names[label])
print("预测类别：", predicted_class, class_names[predicted_class])
print(f"预测类别对应的概率：{predicted_probability:.2%}")
print("所有类别的概率：", probabilities.cpu())