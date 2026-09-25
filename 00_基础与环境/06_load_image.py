#本文件创建一张小型RGB图片，并将图片读取为pytorch张量
#用于观察图片尺寸、张量形状、数据类型和像素值范围

from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

data_directory=Path("00_基础与环境/data")
data_directory.mkdir(parents=True,exist_ok=True)

image_path=data_directory/"sample_orange.png"

created_image=Image.new(
    mode="RGB",
    size=(5,4),
    color=(255,128,0),
)
created_image.save(image_path)

loaded_image=Image.open(image_path).convert("RGB")
image_tensor=pil_to_tensor(loaded_image)

print("image_path:",image_path)
print("PIL image size",loaded_image.size)
print("PIL iamge mode",loaded_image.mode)
print("tensor shape:",image_tensor.shape)
print("tensor dtype:",image_tensor.dtype)
print("minimum value:",image_tensor.min().item())
print("maximum value:",image_tensor.max().item())
print("one pixel:",image_tensor[:,0,0])

float_tensor = image_tensor.to(dtype=torch.float32)/255.0

print("float tensor dtype:",float_tensor.dtype)
print("float minimum:",float_tensor.min().item())
print("float maximum:",float_tensor.max().item())
print("normalized pixel:",float_tensor[:,0,0])