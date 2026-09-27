#学习使用Dataloader将单个样本组合成批次
#观察批次形状，并理解batch_size,shuffle和迭代器作用

import torch
from torch.utils.data import Dataset,DataLoader


class SimpleClassificationDataset(Dataset):
    def __init__(self):
        super().__init__()

        self.features = torch.tensor([
            [1.0, 2.0],
            [2.0, 1.0],
            [-1.0, -2.0],
            [-2.0, -1.0],
        ], dtype=torch.float32)

        self.labels = torch.tensor(
            [0, 0, 1, 1],
            dtype=torch.long,
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        features = self.features[index]
        label = self.labels[index]
        return features, label


dataset = SimpleClassificationDataset()

dataloader=DataLoader(
    dataset,
    batch_size=2,
    shuffle=False,
    num_workers=0                 )


print("样本总数：", len(dataset))
print("批次总数：", len(dataloader))

print("\n使用 for 循环逐批读取：")
for batch_index, (batch_features, batch_labels) in enumerate(dataloader):
    print("\n批次编号：", batch_index)
    print("本批特征：")
    print(batch_features)
    print("特征形状：", batch_features.shape)
    print("本批标签：", batch_labels)
    print("标签形状：", batch_labels.shape)

print("\n使用迭代器手动读取：")
data_iterator=iter(dataloader)

first_features,first_labels=next(data_iterator)
second_features, second_labels = next(data_iterator)

print("第一批特征：")
print(first_features)
print("第一批标签：", first_labels)

print("第二批特征：")
print(second_features)
print("第二批标签：", second_labels)
