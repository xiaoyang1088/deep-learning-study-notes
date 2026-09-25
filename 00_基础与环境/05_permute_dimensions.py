# 本文件使用 permute 调整图像张量的维度顺序。
# 用于理解 HWC 与 CHW 两种图像表示方式。

import torch

image_hwc=torch.rand(4,5,3)

image_chw=image_hwc.permute(2,0,1)

print("HWC shape:",image_hwc.shape)
print("CHW shape:",image_chw.shape)

row=1
column=2

pixel_hwc = image_hwc[row,column,:]
pixel_chw=image_chw[:,row,column]

print("pixel in HWC:",pixel_hwc)
print("pixel in CHW:",pixel_chw)
print("pixel values unchanged:",torch.equal(pixel_hwc,pixel_chw))