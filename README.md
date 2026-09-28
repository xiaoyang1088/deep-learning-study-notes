# Deep Learning Study Notes

本仓库记录我从基础开始学习深度学习的过程，包括环境搭建、PyTorch 基础、论文阅读、代码复现和实验总结。

当前计划依次学习：

1. 深度学习与 PyTorch 基础
2. ResNet
3. Vision Transformer（ViT）
4. CLIP
5. 模型对比与综合复盘

## 当前进度

- [x] 配置 WSL 2 与 Ubuntu
- [x] 创建独立 Conda 学习环境
- [x] 安装支持 CUDA 的 PyTorch
- [x] 验证 NVIDIA GPU 张量计算
- [x] 学习张量、形状和设备
- [x] 理解线性层、损失函数和自动求导
- [x] 使用 SGD 完成最小线性回归训练
- [x] 学习 ReLU、自定义模块与顺序模型
- [x] 理解 NumPy 与 Tensor 转换、共享内存
- [x] 学习 logits、softmax 与类别预测
- [x] 学习交叉熵损失与整数类别标签
- [x] 使用 Dataset 和 DataLoader 组织数据与批次
- [x] 完成小型分类数据的训练与验证流程
- [x] 学习二维卷积、步幅、填充、池化与展平
- [x] 组合最小 CNN，追踪各层的实际数值与张量形状
- [ ] 完成最小图像分类实验
- [ ] 学习并复现 ResNet
- [ ] 学习并复现 Vision Transformer
- [ ] 学习并复现 CLIP

详细进度见[学习进度.md](学习进度.md)。

## 当前环境

- 操作系统：Windows + WSL 2
- Linux：Ubuntu 22.04 LTS
- Python：3.11
- PyTorch：2.13.0
- CUDA：13.0
- GPU：NVIDIA GeForce RTX 5060 Laptop GPU
- Conda 环境：`dl-foundation`

## 目录结构

```text
.
├── 00_基础与环境/
├── 01_ResNet/
├── 02_Vision_Transformer/
├── 03_CLIP/
├── 04_综合复盘与汇报/
├── 共享资源/
├── 学习总路线.md
├── 学习进度.md
└── 项目记忆.md
```

## 下一步

从实验28开始，学习读取图像数据集，观察图像、标签和批次形状，为后续图像分类训练做准备。
