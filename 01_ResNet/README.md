# ResNet：模型实现与训练实验

本目录记录ResNet（Residual Network，残差网络）的论文学习、代码实现和训练实验。从BN、残差相加和基本残差块开始，逐步搭建ResNet18、ResNet34、ResNet50，最后完成ImageNet自选100类训练及CIFAR-10网络深度对比。

**ResNet阶段已结束，以实验21、22作为本阶段的综合实验。原计划中尚未开展的后续实验不再继续。**

## 完整训练成果

三个模型均自行编写，使用随机初始化，不加载预训练权重。

| 实验 | 模型 | 数据集 | 训练规模 | 测试准确率 | 训练耗时 |
|---|---|---|---|---:|---|
| 21 | ResNet50 v1.5 | ImageNet自选100类 | 100轮 | **81.69%** | 约5小时1分27秒，含逐轮验证 |
| 22 | ResNet20 | CIFAR-10 | 64000次参数更新 | **91.71%** | 36.71分钟 |
| 22 | ResNet56 | CIFAR-10 | 64000次参数更新 | **93.15%** | 62.63分钟 |

### 实验21：ResNet50分类训练

实现1×1、3×3、1×1卷积组成的瓶颈残差块，并按3、4、6、3个块组装四个阶段。采用v1.5的下采样步幅位置，100类模型共有23712932个可学习参数。训练前通过与torchvision参考实现的前向对照及梯度、参数更新检查。

固定100类图片经检查和重复清理后，训练、验证、测试分别为116047张、12868张、4993张。在单张A100 40GB上使用批次128训练100轮，每轮验证并保存最佳模型。最终使用第96轮的 `best.pt` 测试，准确率为81.69%。

- [完整代码](21_resnet50)
- [实验报告与运行方式](21_resnet50/report.md)
- [100轮训练记录](21_resnet50/outputs/resnet50_run01/training.csv)
- [最终测试结果](21_resnet50/outputs/resnet50_run01/result.txt)
- [100类标签对应表](21_resnet50/metadata/classes.json)

### 实验22：CIFAR ResNet20与ResNet56对比

参考原论文第4.2节实现CIFAR ResNet，采用三个残差阶段和Option A无参数捷径。两种模型使用相同训练安排：批次128，SGD动量0.9、权重衰减0.0001，在完成32000、48000次更新后降低学习率，64000次更新结束后进行最终测试。

训练使用全部50000张CIFAR-10训练图片，测试使用10000张官方测试图片。在本机RTX 5060 Laptop GPU上完成训练，ResNet20和ResNet56的测试准确率分别为91.71%和93.15%。本次56层模型提高1.44个百分点，训练多用25.92分钟。

- [完整代码](22_cifar10_resnet_training)
- [实验报告与运行方式](22_cifar10_resnet_training/report.md)
- ResNet20：[训练记录](22_cifar10_resnet_training/outputs/22_resnet20/training.csv)、[测试结果](22_cifar10_resnet_training/outputs/22_resnet20/result.txt)
- ResNet56：[训练记录](22_cifar10_resnet_training/outputs/22_resnet56/training.csv)、[测试结果](22_cifar10_resnet_training/outputs/22_resnet56/result.txt)

两个实验的数据集和分类任务不同，测试准确率不直接横向比较。具体环境、数据处理和模型设置见各自报告。

## 基础实验目录

以下为已整理到仓库的脚本，按学习顺序保留。基础实验用于理解局部计算与训练流程，完整训练结果以上方两个综合实验为准。

| 编号 | 代码 | 内容 |
|---|---|---|
| 01 | [environment and review](01_environment_and_review.py) | 环境检查与基础复习，查看软件版本、设备和张量计算 |
| 03 | [first forward](03_resnet18_first_forward.py) | 首次前向追踪，查看ResNet18各阶段形状并拆开残差块计算 |
| 04 | [batch norm](04_batch_norm.py) | 批归一化，对比训练与评估模式及运行统计量 |
| 05 | [residual addition](05_residual_addition.py) | 残差相加，观察零残差和ReLU的影响 |
| 06 | [basic block](06_basic_block.py) | 基本残差块，自定义保持形状的两层卷积块 |
| 07 | [projection shortcut](07_projection_shortcut.py) | 投影捷径，处理通道或空间尺寸变化 |
| 08 | [build ResNet18](08_build_resnet18.py) | 组装完整ResNet18，与参考实现核对 |
| 09 | [debug ResNet18](09_debug_resnet18.py) | 调试模型，观察前向流程与局部变量 |
| 10 | [CIFAR-10 data](10_cifar10_data.py) | 数据读取、样本查看、训练验证划分与批次组织 |
| 11 | [one batch training](11_one_batch_training.py) | 单批次训练，检查损失、梯度与参数更新 |
| 12 | [overfit small batch](12_overfit_small_batch.py) | 小批次拟合，检查模型能否学会固定的少量图片 |
| 13 | [train and validate](13_train_and_validate.py) | 训练与验证，保存指标和模型 |
| 14 | [load and predict](14_load_and_predict.py) | 加载与预测，复核验证结果并查看样本 |
| 15 | [learning rate comparison](15_learning_rate_comparison.py) | 学习率对照，观察不同学习率的训练表现 |
| 16 | [data augmentation](16_data_augmentation.py) | 数据增强，对比随机裁剪、翻转等处理 |
| 19 | [build ResNet34](19_build_resnet34.py) | 组装ResNet34，理解阶段块数与网络深度 |
| 20 | [bottleneck block](20_bottleneck_block.py) | 瓶颈残差块，对照恒等和变换捷径及参考实现 |

编号02、17对应论文阅读，没有单独列出代码；编号18未形成独立脚本。本目录保留原编号，不因后续学习安排变化而重新编号。实验21、22以实际代码目录和报告内容为准，已替代早期计划中的同编号主题。

## 运行说明

默认使用项目首页列出的本机WSL Ubuntu与 `dl-foundation` 环境。下面的基础实验示例在仓库根目录执行：

```bash
conda activate dl-foundation
python 01_ResNet/03_resnet18_first_forward.py
```

实验09使用交互式调试，出现 `(Pdb)` 表示程序暂停等待调试命令。实验09、11～14会复用实验08的模型定义；部分脚本中的数据和模型路径需按实际保存位置设置。实验14需要先准备实验13保存的模型。

实验21和22应进入各自目录，按报告中的命令运行。正式训练入口均要求CUDA，输出目录应为新目录。数据需提前准备：实验21依赖图片、类别表及固定清单；实验22直接读取已有CIFAR-10，训练代码不自动下载。

## 训练权重与结果文件

**[下载实验21、22训练权重](https://github.com/xiaoyang1088/deep-learning-study-notes/releases/tag/resnet_experiments_v1)**

| Release附件 | 内容 | 下载后的本地位置，相对于本目录 |
|---|---|---|
| `best.pt` | 实验21第96轮最佳模型，含模型状态、类别表等信息 | `21_resnet50/outputs/resnet50_run01/best.pt` |
| `resnet20.pt` | 实验22的20层模型状态，含参数及BN统计量 | `22_cifar10_resnet_training/outputs/22_resnet20/resnet20.pt` |
| `resnet56.pt` | 实验22的56层模型状态，含参数及BN统计量 | `22_cifar10_resnet_training/outputs/22_resnet56/resnet56.pt` |

数据集和图片打包文件不纳入仓库；权重通过Release提供，克隆仓库不会自动下载它们。实验21、22的训练CSV和结果文本已经提交到对应 `outputs` 目录，可直接查看。

## 参考资料

He K, Zhang X, Ren S, Sun J. Deep Residual Learning for Image Recognition. CVPR, 2016: 770–778. [论文页面](https://arxiv.org/abs/1512.03385)。模型实现对照torchvision中的ResNet。
