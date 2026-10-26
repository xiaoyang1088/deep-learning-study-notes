# Deep Learning Study Notes

本仓库记录我学习深度学习的过程，包括PyTorch基础、论文阅读、模型实现、训练测试和实验报告。学习顺序为模型基础、ResNet、Vision Transformer（ViT）和CLIP，通过自己编写代码和实际训练理解模型。

ResNet阶段已完成，以实验21、22的完整训练与测试结束本阶段，不再继续原计划中的后续ResNet实验。目前开始学习ViT。

## 主要实验成果

以下三个模型均从随机初始化开始训练，不使用预训练权重。

| 实验 | 模型 | 数据集 | 训练规模 | 测试准确率 | 训练耗时 |
|---|---|---|---|---:|---|
| 21 | ResNet50 v1.5 | ImageNet自选100类 | 100轮 | **81.69%** | 约5小时1分27秒，含逐轮验证 |
| 22 | ResNet20 | CIFAR-10 | 64000次参数更新 | **91.71%** | 36.71分钟 |
| 22 | ResNet56 | CIFAR-10 | 64000次参数更新 | **93.15%** | 62.63分钟 |

实验21使用116047张训练图片、12868张验证图片和4993张测试图片，最终测试使用第96轮的最佳验证模型。实验22使用CIFAR-10的50000张训练图片和10000张测试图片，比较20层与56层网络；本次56层模型的测试准确率提高1.44个百分点。两个实验的分类任务不同，准确率不直接横向比较。

| 内容 | 实验21 | 实验22 |
|---|---|---|
| 代码 | [ResNet50](01_ResNet/21_resnet50) | [CIFAR ResNet20与ResNet56](01_ResNet/22_cifar10_resnet_training) |
| 实验报告 | [报告](01_ResNet/21_resnet50/report.md) | [报告](01_ResNet/22_cifar10_resnet_training/report.md) |
| 完整训练记录及结果 | [100轮记录](01_ResNet/21_resnet50/outputs/resnet50_run01) | [20层记录](01_ResNet/22_cifar10_resnet_training/outputs/22_resnet20)、[56层记录](01_ResNet/22_cifar10_resnet_training/outputs/22_resnet56) |

**[下载训练权重](https://github.com/xiaoyang1088/deep-learning-study-notes/releases/tag/resnet_experiments_v1)**：`best.pt`为实验21第96轮模型，`resnet20.pt`和`resnet56.pt`为实验22训练结束后的模型。使用时按对应实验的模型结构和预处理加载。

## 学习进度

| 阶段 | 状态 | 已完成内容或当前方向 |
|---|---|---|
| PyTorch与图像分类基础 | 已完成基础实践 | 张量、自动求导、优化器、卷积、数据加载、训练测试及模型保存 |
| ResNet | 已完成，阶段结束 | 论文阅读、残差块与网络实现、训练对照、实验21与22完整训练测试 |
| Vision Transformer | 学习中 | 从图像分块与序列表示开始，学习原论文及模型实现 |
| CLIP | 待开始 | 后续学习图文编码器、对比学习和零样本分类 |

## ResNet阶段

从卷积、BN与残差相加开始，逐步实现基本残差块、投影捷径、ResNet18、ResNet34和ResNet50；随后学习数据增强、学习率与训练评估，并完成两个综合实验。实验21训练100类自然图片分类模型，实验22参考论文的CIFAR网络结构和训练安排，完成网络深度对比。

具体实现、运行方式和分析见上方代码与报告。其他基础实验见 [ResNet目录](01_ResNet)。

## 运行环境

本项目默认在本机Windows + WSL 2的Ubuntu环境中编写和运行代码，包括PyTorch基础、ResNet各项本地实验及实验22训练。实验21的ResNet50完整训练使用矩池云A100，单独列出其环境。

| 项目 | 本机默认环境 | 实验21云端训练环境 |
|---|---|---|
| 用途 | 日常代码学习、运行与调试；基础实验、ResNet本地实验和实验22训练 | 实验21的ResNet50 v1.5完整训练与测试 |
| 系统 | Windows + WSL 2，Ubuntu 22.04 | Ubuntu 22.04 |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU，单卡 | NVIDIA A100-PCIE-40GB，单卡 |
| 显存 | 8GB | 40GB |
| Python | 3.11.16 | 3.11 |
| PyTorch | 2.13.0+cu130 | 2.6.0+cu124 |
| torchvision | 0.28.0+cu130 | 0.21.0+cu124 |
| PyTorch对应的CUDA运行时 | 13.0 | 12.4 |
| Conda环境 | `dl-foundation` | `myconda` |

表中CUDA版本指PyTorch对应的运行时版本。本机版本列出当前学习环境及实验22使用的配置；早期实验如有版本差异，以对应记录为准。具体训练设置和运行命令见各实验报告。数据集不包含在仓库中；模型权重通过Release单独下载。

## 目录

| 目录 | 内容 |
|---|---|
| [00_基础与环境](00_基础与环境) | 环境搭建与PyTorch基础实践 |
| [01_ResNet](01_ResNet) | ResNet基础实验、完整训练代码、报告和结果 |
| [02_Vision_Transformer](02_Vision_Transformer) | ViT学习与实验 |
| [03_CLIP](03_CLIP) | CLIP后续学习目录 |
| [04_综合复盘与汇报](04_综合复盘与汇报) | 综合复盘与汇报目录 |
| [共享资源](共享资源) | 共享学习资源 |

后续继续学习ViT，再学习CLIP。
