# 00 基础与环境

本目录记录深度学习环境搭建、PyTorch 张量基础和图像数据处理实验。

## 当前环境

- Ubuntu 22.04 LTS（WSL 2）
- Python 3.11
- PyTorch 2.13.0
- CUDA 13.0
- NVIDIA GeForce RTX 5060 Laptop GPU
- Conda 环境：`dl-foundation`

## 已完成实验

| 文件 | 内容 |
| --- | --- |
| `01_tensor_shape.py` | 张量的形状、维度、数据类型和设备 |
| `02_image_tensor.py` | RGB 图像张量与像素 |
| `03_batch_tensor.py` | NCHW 批次张量 |
| `04_batch_dimension.py` | 增加和移除批次维度 |
| `05_permute_dimensions.py` | HWC 与 CHW 维度转换 |
| `06_load_image.py` | 从图片文件读取 PyTorch 张量 |
| `07_compare_image_conversion.py` | 比较两种图片转张量方法 |
| `08_channel_normalization.py` | 通道标准化与广播 |
| `09_matrix_multiplication.py` | 逐元素乘法与矩阵乘法 |

## 运行方法

进入项目目录并激活环境：

```bash
conda activate dl-foundation
cd /mnt/c/Users/34138/Desktop/深度学习
```

运行单个实验：

```bash
python 00_基础与环境/01_tensor_shape.py
```

将文件名替换为其他实验文件，即可运行对应内容。

## 当前理解

- PyTorch 张量使用 `shape` 表示各维度大小，使用 `ndim` 表示维度数量。
- 单张彩色图像通常使用 `[C,H,W]`，批次图像使用 `[N,C,H,W]`。
- `unsqueeze`、`squeeze` 和 `permute` 用于调整张量维度。
- 普通图片通常使用 `uint8` 和 `0～255`，模型输入通常转换为 `float32`。
- 广播可以让不同但兼容的形状共同参与计算。
- `*` 表示逐元素乘法，`@` 表示矩阵乘法。

## 下一步

学习 PyTorch 线性层、损失函数、自动求导和参数更新。
