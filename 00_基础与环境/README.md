# 00 基础与环境

本目录记录深度学习环境搭建、PyTorch 张量基础和图像数据处理实验。

详细实验安排见 [`阶段0实验计划.md`](阶段0实验计划.md)。

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
| `10_linear_layer.py` | `nn.Linear` 与手工矩阵计算 |
| `11_mse_loss.py` | 均方误差损失函数 |
| `12_autograd.py` | 自动求导与参数梯度 |
| `13_manual_parameter_update.py` | 手工更新权重和偏置 |
| `14_optimizer_training_loop.py` | SGD 优化器与训练循环 |
| `15_linear_regression.py` | 最小线性回归训练 |
| `16_relu_activation.py` | ReLU 激活函数与绝对值网络 |
| `17_custom_module.py` | 使用自定义 nn.Module 组织网络 |
| `18_numpy_tensor_conversion.py` | NumPy 与 Tensor 转换、共享内存与复制 |
| `19_sequential_model.py` | 使用 nn.Sequential 顺序组织网络 |
| `20_logits_softmax_argmax.py` | 分类分数、概率与预测类别 |

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
- `nn.Linear` 使用权重和偏置完成 `y=xWᵀ+b`。
- 损失函数用于衡量预测值与目标值的差距。
- `backward()` 计算梯度，梯度保存在参数的 `.grad` 中。
- 优化器根据梯度更新参数，训练循环会重复前向计算、反向传播和参数更新。
- ReLU 计算 max(0, x)，不是绝对值；ReLU(x) + ReLU(-x) 才能得到绝对值。
- 自定义 nn.Module 通过 forward() 定义计算过程，nn.Sequential 按给定顺序运行各个模块。
- copy_() 将数值复制到已有张量中，末尾的下划线表示原地修改。
- torch.from_numpy() 创建的张量与原数组共享内存，torch.tensor() 会复制数据。
- 本次实验中，CPU 张量调用 numpy() 得到的数组与张量共享内存。
- logits 是原始分类分数，不是概率；softmax 将每个样本的类别分数转换为概率。
- 对形状为 [样本数, 类别数] 的输出，dim=1 表示沿类别维度计算。
- argmax 返回最大值的位置，用于得到预测类别编号。
- enumerate() 提供位置编号和元素，两个循环变量通过解包按顺序接收它们。

## 下一步

学习实验 21：交叉熵损失与整数类别标签，理解如何衡量分类预测与真实类别之间的差距。

## 实验 16～20 的结果对照

以下结果对应脚本中的固定输入与参数，可通过运行各脚本复核。

| 实验 | 结果 |
| --- | --- |
| 16 | 不使用 ReLU 时输出全为 0；使用 ReLU 后得到输入的绝对值 |
| 17、19 | 自定义模块和 Sequential 均得到 `[2, 1, 0, 1, 2]`，输出形状为 `[5, 1]` |
| 18 | 修改共享数据会影响原数组和共享张量，复制得到的张量保持原值 |
| 20 | 两行概率分别约为 `[0.6652, 0.2447, 0.0900]` 和 `[0.0420, 0.1142, 0.8438]`，预测为猫和鸟；每行概率之和约为 1 |
