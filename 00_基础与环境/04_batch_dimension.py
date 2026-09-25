# 本文件使用 unsqueeze 和 squeeze 增加或移除批次维度。
# 用于理解单张图像如何转换为模型需要的批次输入。

import torch

image = torch.rand(3,4,5)

batch= image.unsqueeze(dim=0)
restored_image=batch.squeeze(dim=0)

print("image shape:",image.shape)
print("batch shape:",batch.shape)
print("restored image shape:",restored_image.shape)
print("values unchanged:",torch.equal(image,restored_image))