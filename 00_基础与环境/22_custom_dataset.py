#学习使用Dataset将输入特征与类别标签组织成数据集
#实现样本数量查询和按索引读取，观察每个样本的形状和数据类型

import torch
from torch.utils.data import Dataset

class SimpleClassificationDataset(Dataset):
    def __init__(self):
        super().__init__()

        self.features=torch.tensor([
            [1.0,2.0],
            [2.0,1.0],
            [-1.0,-2.0],
            [-2.0,-1.0]
        ],
        dtype=torch.float32
        )

        self.labels=torch.tensor(
            [0,0,1,1],
            dtype=torch.long
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        features=self.features[index]
        label=self.labels[index]
        return features,label


dataset=SimpleClassificationDataset()

print("样本总数：",len(dataset))

sample_features,sample_label=dataset[0]

print("\n第一个样本的特征：",sample_features)
print("特征形状：", sample_features.shape)
print("特征数据类型：", sample_features.dtype)

print("\n第一个样本的标签：")
print(sample_label)
print("标签形状：", sample_label.shape)
print("标签数据类型：", sample_label.dtype)

print("\n逐个读取所有样本：")
for sample_index in range (len(dataset)):
    features,label=dataset[sample_index]
    print(
        "样本编号：",sample_index,
        "特征：",features.tolist(),
        "标签：",label.item()
    )