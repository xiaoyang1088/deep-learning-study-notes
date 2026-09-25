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
