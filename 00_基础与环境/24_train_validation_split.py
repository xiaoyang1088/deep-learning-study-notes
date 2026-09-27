#将小型数据集划分为训练集和验证集
#练习按批次训练，并区分train(),eval()和no_grad()的作用

import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader,random_split


class SimpleClassificationDataset(Dataset):
    def __init__(self):
        super().__init__()

        self.features = torch.tensor([
            [1.0, 2.0],
            [2.0, 1.0],
            [2.0, 3.0],
            [3.0, 2.0],
            [-1.0, -2.0],
            [-2.0, -1.0],
            [-2.0, -3.0],
            [-3.0, -2.0],
        ], dtype=torch.float32)

        self.labels = torch.tensor(
            [0, 0, 0, 0, 1, 1, 1, 1],
            dtype=torch.long,
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]

torch.manual_seed(0)

dataset=SimpleClassificationDataset()

train_dataset,validation_dataset=random_split(
    dataset,
    [6,2],
    generator=torch.Generator().manual_seed(0)
)

train_loader= DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True,
    num_workers=0
)

validation_loader=DataLoader(
    validation_dataset,
    batch_size=2,
    shuffle=False,
    num_workers=0
)

model=nn.Linear(in_features=2,out_features=2)

loss_function=nn.CrossEntropyLoss()

optimizer=torch.optim.SGD(model.parameters(),lr=0.1)

print("训练集样本数：", len(train_dataset))
print("验证集样本数：", len(validation_dataset))
print("训练集原始索引：", train_dataset.indices)
print("验证集原始索引：", validation_dataset.indices)

for epoch in range(10):
    model.train()
    training_loss_sum=0.0

    for batch_features,batch_labels in train_loader:
        optimizer.zero_grad()

        logits = model(batch_features)
        loss=loss_function(logits,batch_labels)

        loss.backward()
        optimizer.step()
        training_loss_sum+=loss.item()*batch_labels.shape[0]
    average_training_loss=training_loss_sum/len(train_dataset)

    model.eval()
    validation_loss_sum=0.0
    correct_count=0
    with torch.no_grad():
        for batch_features,batch_labels in validation_loader:
            logits=model(batch_features)
            loss=loss_function(logits,batch_labels)
            validation_loss_sum+=loss.item()*batch_labels.shape[0]

            prediction=torch.argmax(logits,dim=1)
            correct_count+=(prediction==batch_labels).sum().item()
    average_validation_loss =(validation_loss_sum / len(validation_dataset))
    validation_accuracy = correct_count / len(validation_dataset)

    print(
    "轮次：", epoch + 1,
    "训练损失：", average_training_loss,
    "验证损失：", average_validation_loss,
    "验证准确率：", validation_accuracy,
    )