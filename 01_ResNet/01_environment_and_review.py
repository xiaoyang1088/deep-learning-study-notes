#检查ResNet学习使用的python/pytorch和计算设备
#创建一个小张量，确认选定设备能实际执行计算

import sys 

import torch
import torchvision

print("python路径：",sys.executable)
print("python版本：",sys.version)
print("pytorch版本：",torch.__version__)
print("torchvision版本：",torchvision.__version__)

cuda_available=torch.cuda.is_available()
print("CUDA是否可用：",cuda_available)

device=torch.device("cuda"if cuda_available else "cpu")
print("本次计算设备：",device)

if cuda_available:
    print("GPU名称：",torch.cuda.get_device_name(0))


values=torch.tensor(
    [1.0,2.0,3.0],
    device=device
)

result=values*2

print("输入张量：",values)
print("计算结果：",result)
print("结果所在设备：",result.device)


#创建两张模拟彩色图片，以及各自对应的类别标签
images=torch.rand(2,3,32,32,device=device)

labels=torch.tensor(
    [1,7],
    dtype=torch.long,
    device=device
)

print("\n图片批次形状：",images.shape)
print("图片数据类型：",images.dtype)
print("标签：",labels)
print("标签形状：",labels.shape)
print("标签数据类型：",labels.dtype)

#后面遇到问题再复习