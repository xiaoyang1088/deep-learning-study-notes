# 实验09:让程序在指定位置暂停，查看当时的变量，再逐句执行

#复用实验08的模型，用调试器观察标准输入版和小图片版的前向计算

import runpy
from pathlib import Path

import torch
from torch import nn

#读取实验08的已有模型定义
model_path=Path(__file__).resolve().parent/"08_build_resnet18.py"
experiment=runpy.run_path(str(model_path))   #字典

BasicBlock=experiment["BasicBlock"]
ResNet18=experiment["ResNet18"]

test_cases=[
    ('标准输入版',224,1000),
    ("小图片版",32,10)
]

for case_name,image_size,num_classes in test_cases:
    model=ResNet18(num_classes=num_classes)

    #调整小图片版的入口：
    if image_size==32:
        model.conv1=nn.Conv2d(
            3,64,kernel_size=3,stride=1,padding=1,bias=False
        )
        # identity是恒等
        model.maxpool=nn.Identity()

    model.eval()
    images=torch.randn(2,3,image_size,image_size)

    print("\n当前情况：",case_name)
    print("输入形状：",images.shape)

    #暂停后设置断点，进入模型和残差块
    with torch.no_grad():
        breakpoint()
        output=model(images)

    print("输出形状：",output.shape)

# p 表达式:打印表达式的结果
# n:执行当前行,停到下一行
# s:执行一步，遇到函数调用时尝试进入
# b 位置:设置断点
# c:继续执行，直到下一次暂停
# l:查看暂停位置附近的代码 

