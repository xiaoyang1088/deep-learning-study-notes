# 模块3：数据准备、从零训练与评估
# 实验10(CIFAR-10 数据准备实验) 准备真实图片和标签，确认它们怎样组成批次

# 读取CIFAR-10,划分训练与验证集，查看样本并检查批次形状。

from pathlib import Path
import torch 
from torch.utils.data import DataLoader,random_split
from torchvision import datasets,transforms
from torchvision.transforms.functional import to_pil_image

#下载并读取官方训练集合测试集
data_dir=Path(__file__).resolve().parent/"data"


transform=transforms.ToTensor()
# 针对这里的 CIFAR-10 图片，它主要做两件事：
# 1. 将图片组织为 [通道,高,宽] 的张量。
# 2. 将原来的整数像素值 0～255 转换为浮点数 0～1。

full_train_dataset=datasets.CIFAR10(
    root=str(data_dir),
    train=True,    #读取官方训练部分
    transform=transform,    #每次取图片时执行的转换
    download=True   #缺少有效数据时下载并解压
)

test_dataset=datasets.CIFAR10(
    root=str(data_dir),
    train=False,
    transform=transform,
    download=True
)

print("官方训练集数量：",len(full_train_dataset))
print("官方测试集数量：",len(test_dataset))
print("类别名称：",full_train_dataset.classes)

#从官方训练集中划分训练集和验证集,设置种子42,是为了后面在相同数据与实现环境下重复运行时,能够复现这次划分
split_generator=torch.Generator().manual_seed(42)

train_dataset,val_dataset=random_split(
    full_train_dataset,
    [45000,5000],
    generator=split_generator
)

print("本次训练集数量：",len(train_dataset))
print("本次验证集数量：",len(val_dataset))

#查看一个训练样本并保存图片

image,label=train_dataset[0]
print("单张图片形状：",image.shape)
print("图片数据类型：",image.dtype)
print("像素最小值：", image.min().item())
print("像素最大值：", image.max().item())
print("标签编号：", label)
print("标签名称：", full_train_dataset.classes[label])

preview_path=data_dir/"cifar10_sample.png"
to_pil_image(image).save(preview_path)
#PIL为python图像库，现在通常使用它的后继库 Pillow
# to_pil_image(image) 将它转换成 Pillow 能处理的图片对象。对于这里的 RGB 图片，会把通道和像素组织成常规图片形式，并将浮点像素转换为 0～255 的整数像素
print("样本图片保存位置：",preview_path)


#将样本组织成批次

train_loader=DataLoader(
    train_dataset,batch_size=64,shuffle=True,num_workers=0,generator=torch.Generator().manual_seed(42)
)
# shuffle=True：打乱取样顺序

val_loader=DataLoader(
    val_dataset,
    batch_size=64,
    shuffle=False,num_workers=0
)

images,labels=next(iter(train_loader))
# iter(train_loader)：创建一个可以逐批取数据的迭代器
# next(...)：取出第一批
val_images,val_labels=next(iter(val_loader))

print("训练图片批次形状：", images.shape)
print("训练标签批次形状：", labels.shape)
print("训练标签数据类型：", labels.dtype)
print("前十个训练标签：", labels[:10])

print("验证图片批次形状：", val_images.shape)
print("验证标签批次形状：", val_labels.shape)