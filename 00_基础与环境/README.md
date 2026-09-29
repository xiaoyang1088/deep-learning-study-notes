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
| `21_cross_entropy_loss.py` | 交叉熵损失、整数标签与手工计算对照 |
| `22_custom_dataset.py` | 自定义 Dataset，按索引读取特征和标签 |
| `23_dataloader.py` | 批次加载、shuffle 与迭代器 |
| `24_train_validation_split.py` | 划分训练集和验证集，完成训练与验证循环 |
| `25_conv2d.py` | 二维卷积、卷积核、步幅、填充与输出形状 |
| `26_pooling_flatten.py` | 最大池化与展平，观察数值和形状变化 |
| `27_simple_cnn.py` | 组合最小 CNN，追踪完整前向传播 |
| `28_image_dataset.py` | 读取 Fashion-MNIST，查看图片、标签与批次形状 |
| `29_train_image_classifier.py` | 训练小型 CNN，进行验证并保存最后一轮参数 |
| `30_evaluate_save_load.py` | 加载参数，评估官方测试集并预测单张图片 |

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

### 图像分类实验运行顺序

在已配置的环境中，从项目根目录依次运行：

```bash
python 00_基础与环境/28_image_dataset.py
python 00_基础与环境/29_train_image_classifier.py
python 00_基础与环境/30_evaluate_save_load.py
```

实验28首次运行会下载Fashion-MNIST数据；实验29训练模型并保存参数；实验30加载参数进行测试与预测。

数据集和权重文件不上传仓库。首次运行需要按上述顺序准备；重新运行实验29会从头训练，并覆盖同名权重文件。

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
- 使用整数类别标签时，CrossEntropyLoss 接收原始 logits 和 torch.long 类型的标签，不需要先计算 softmax。
- Dataset 的 __len__() 返回样本数量，__getitem__() 按索引返回特征和标签。
- batch_size 表示每批包含的样本数量，与每个样本的特征数量不同。
- DataLoader 将单个样本组合成批次；shuffle 改变读取顺序，但不破坏特征与标签的对应关系。
- iter() 创建迭代器，next() 获取下一批数据。
- backward() 根据链式法则计算梯度，optimizer.step() 使用梯度更新参数。
- 梯度默认会累加，因此本实验在每批训练前使用 optimizer.zero_grad() 清除旧梯度。
- 每轮训练完整遍历一次训练集，下一轮继续使用上一轮更新后的参数。
- model.train() 和 model.eval() 切换运行模式，不会直接更新参数。
- no_grad() 关闭代码块内的梯度记录，避免为验证计算建立反向传播所需的计算图。
- 验证阶段不执行 backward() 和 optimizer.step()，本实验的验证结果不会自动改变下一轮训练。
- 平均损失按样本数量计算，准确率为预测正确的样本数除以样本总数。




