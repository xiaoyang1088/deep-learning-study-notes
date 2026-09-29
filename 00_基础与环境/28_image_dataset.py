#下载并读取Fashion-MNIST，观察单张图像、类别标签和批次张量
#将一个样本保存为图片，方便查看；本实验不进行模型训练

from pathlib import Path 

from torch.utils.data import DataLoader
from torchvision import datasets,transforms
from torchvision.transforms.functional import to_pil_image

experiment_dir=Path(__file__).resolve().parent
data_dir=experiment_dir/"data"

image_transforms=transforms.ToTensor()

train_dataset=datasets.FashionMNIST(
    root=str(data_dir),
    train=True,
    transform=image_transforms,
    download=True
)

class_names=[
    "T恤或上衣",
    "裤子",
    "套头衫",
    "连衣裙",
    "外套",
    "凉鞋",
    "衬衫",
    "运动鞋",
    "包",
    "短靴"]

image,label =train_dataset[0]

print("训练集样本数：", len(train_dataset))
print("单张图像形状：", image.shape)
print("图像数据类型：", image.dtype)
print("像素最小值：", image.min().item())
print("像素最大值：", image.max().item())
print("标签：", label)
print("标签的 Python 类型：", type(label))
print("类别名称：", class_names[label])

preview_path=data_dir/"sample_0.png"
to_pil_image(image).save(preview_path)
print("样本图片保存位置：",preview_path)

train_loader=DataLoader (
    train_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=0
)

batch_images,batch_labels =next(iter(train_loader))

print("\n一批图像的形状：", batch_images.shape)
print("一批标签的形状：", batch_labels.shape)
print("一批标签的数据类型：", batch_labels.dtype)
print("一批标签：", batch_labels)

for sample_index, class_index in enumerate(batch_labels.tolist()):
    print("批次内样本编号：", sample_index, "类别：", class_names[class_index])