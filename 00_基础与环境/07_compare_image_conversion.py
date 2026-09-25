#本文件比较PIL图片转化为pytorch张量的两种方法
#用于重点观察转换后的数据类型、形状和像素值范围

from pathlib import Path

from PIL import Image
from torchvision.transforms.functional import pil_to_tensor,to_tensor

image_path=Path("00_基础与环境/data/sample_orange.png")
image=Image.open(image_path)

integer_tensor=pil_to_tensor(image)
float_tensor=to_tensor(image)

print("image mode:",image.mode)
print("image size:",image.size)

print("pil_to_tensor shape:", integer_tensor.shape)
print("pil_to_tensor dtype:", integer_tensor.dtype)
print("pil_to_tensor pixel:", integer_tensor[:, 0, 0])
print("pil_to_tensor range:", integer_tensor.min().item(), integer_tensor.max().item())

print("to_tensor shape:", float_tensor.shape)
print("to_tensor dtype:", float_tensor.dtype)
print("to_tensor pixel:", float_tensor[:, 0, 0])
print("to_tensor range:", float_tensor.min().item(), float_tensor.max().item())