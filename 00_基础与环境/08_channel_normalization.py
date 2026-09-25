#本文件对RGB图像的三个颜色通道分别进行标准化
#用于理解均值、标准差、reshape和pytorch广播机制

from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

image_path=Path("00_基础与环境/data/sample_orange.png")
image=to_tensor(Image.open(image_path))

channel_mean=torch.tensor([0.5,0.5,0.5]).reshape(3,1,1)
channel_std = torch.tensor([0.5, 0.5, 0.5]).reshape(3, 1, 1)

normalized_image=(image-channel_mean)/channel_std
restored_image=normalized_image*channel_std+channel_mean

print("image shape:", image.shape)
print("mean shape:", channel_mean.shape)
print("std shape:", channel_std.shape)

print("original pixel:", image[:, 0, 0])
print("normalized pixel:", normalized_image[:, 0, 0])
print("restored pixel:", restored_image[:, 0, 0])

print("restored correctly:",torch.allclose(image,restored_image))