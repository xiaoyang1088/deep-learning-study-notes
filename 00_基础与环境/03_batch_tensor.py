# 本文件把多张图像组合成一个四维批次张量。
# 用于理解神经网络输入的 NCHW 形状。

import torch

black_image=torch.zeros(3,4,5)
white_image=torch.ones(3,4,5)

batch=torch.stack(
    [black_image,white_image],
    dim=0
)

print("single image shape:",black_image.shape)
print("batch shape:",batch.shape)
print("batch ndim:",batch.ndim)
print("number of values:",batch.numel())

print("first image shape:",batch[0].shape)
print("first image pixel:",batch[0,:,1,2])
print("second image pixel:",batch[1,:,1,2])

print("original device:",batch.device)

batch_gpu=batch.to("cuda")

print("GPU device:",batch_gpu.device)