#创建未加载预训练权重的ResNet18
#使用模拟彩色图片进行一次前向计算，查看模型结构和输入输出形状

import torch 
from torchvision.models import resnet18

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model=resnet18(weights=None)
model=model.to(device)
model.eval()

images=torch.rand(2,3,224,224,device=device)

print("计算设备：",device)
print("模型结构：",model)

with torch.no_grad():
    logits=model(images)


print("\n输入形状：",images.shape)
print("输出形状：",logits.shape)
print("第一张图片的前10个分类分数：",logits[0,:10])

#单独观察ResNet18入口卷积的权重和输入输出形状
with torch.no_grad():
    conv1_output=model.conv1(images)

print("\n入口卷积：",model.conv1)
print("卷积权重形状：", model.conv1.weight.shape)
print("卷积输入形状：", images.shape)
print("卷积输出形状：", conv1_output.shape)

#观察入口卷积之后的BN层，比较处理前后的形状和部分数值
with torch.no_grad():
    bn1_output=model.bn1(conv1_output)

print("\n入口BN：",model.bn1)
print("BN 输入形状：", conv1_output.shape)
print("BN 输出形状：", bn1_output.shape)

print("BN 前的部分数值：", conv1_output[0, 0, 0, :5])
print("BN 后的部分数值：", bn1_output[0, 0, 0, :5])


#观察ReLU如何将负数变为0，使用副本保留原始BN输出
with torch.no_grad():
    relu_output=model.relu(bn1_output.clone())



print("\n入口激活函数：", model.relu)
print("ReLU 输入形状：", bn1_output.shape)
print("ReLU 输出形状：", relu_output.shape)

print("ReLU 前的部分数值：", bn1_output[0, 0, 0, :5])
print("ReLU 后的部分数值：", relu_output[0, 0, 0, :5])

print("ReLU 前的最小值：", bn1_output.min().item())
print("ReLU 后的最小值：", relu_output.min().item())


# 观察最大池化如何缩小特征图，并核对一个输出位置的计算
#最大池化maxpool,它在特征图的每个小区域里选出最大值，用这些最大值组成一张更小的特征图。这里会将宽高从 112×112 缩到 56×56，减少后续计算，但也会丢掉一部分细节。
#PyTorch 最大池化的边界填充可以理解为负无穷

with torch.no_grad():
    maxpool_output=model.maxpool(relu_output)

print("\n入口最大池化：", model.maxpool)
print("最大池化输入形状：", relu_output.shape)
print("最大池化输出形状：", maxpool_output.shape)

print("输入中选取的 3×3 区域：")
print(relu_output[0, 0, 1:4, 1:4])
print("该区域的最大值：", relu_output[0, 0, 1:4, 1:4].max().item())
print("对应的池化输出值：", maxpool_output[0, 0, 1, 1].item())



# layer1，也就是第一个残差阶段。它会继续加工池化后的特征，但不是只让数据经过卷积：每个残差块还会把输入直接传到后面，与卷积分支的结果相加

#按顺序运行第一残差阶段的两个基本残差块，观察形状变化

with torch.no_grad():
    layer1_block0_output=model.layer1[0](maxpool_output)
    layer1_output=model.layer1[1](layer1_block0_output)

print("\n第一残差阶段结构：", model.layer1)
print("第一残差阶段输入形状：", maxpool_output.shape)
print("第一个残差块输出形状：", layer1_block0_output.shape)
print("第二个残差块输出形状：", layer1_output.shape)

#拆开第一个残差块，核对F(x)+x以及最终输出
#先算卷积分支的F(x)，再加回原输入x，最后经过 ReLU
block=model.layer1[0]
block_input=maxpool_output

with torch.no_grad():
    branch_output=block.conv1(block_input)
    branch_output=block.bn1(branch_output)
    branch_output=block.relu(branch_output)
    branch_output=block.conv2(branch_output)
    residual_output=block.bn2(branch_output)

    sum_output=residual_output+block_input
    manual_output=block.relu(sum_output.clone())

print("\n原输入 x 的形状：", block_input.shape)
print("残差 F(x) 的形状：", residual_output.shape)
print("相加后的形状：", sum_output.shape)
print("最终输出形状：", manual_output.shape)

print("\n原输入 x 的部分数值：", block_input[0, 0, 0, :5])
print("残差 F(x) 的部分数值：", residual_output[0, 0, 0, :5])
print("F(x) + x 的部分数值：", sum_output[0, 0, 0, :5])
print("经过 ReLU 后：", manual_output[0, 0, 0, :5])

print("检查拆开计算与直接调用是否一致：",torch.allclose(manual_output,layer1_block0_output))


# layer2，即第二个残差阶段。这里会把特征图从 64 个通道的 56×56，变成 128 个通道的 28×28
#改变形状的是第一个块layer2[0]，结果为ReLU(F(x)+P(x))，P(x)代表变换后的捷径输出，第二个块保持形状不变。

#观察第二残差阶段的形状变化，以及第一个块的捷径变换
#第一个块的捷径变化没有完整保留x，他的stride=2

with torch.no_grad():
    layer2_shortcut_output=model.layer2[0].downsample(layer1_output)#这句代码只是为了展示layer2[0]的变换捷径，他是经过卷积和BN，形状和数值都被处理，不保证原值不变，不是直接复制输入
    layer2_block0_output=model.layer2[0](layer1_output)
    layer2_output=model.layer2[1](layer2_block0_output)

print("\n第二残差阶段的第一个块：", model.layer2[0])
print("第二残差阶段输入形状：", layer1_output.shape)
print("第一个块的捷径输出形状：", layer2_shortcut_output.shape)
print("第一个残差块输出形状：", layer2_block0_output.shape)
print("第二个残差块输出形状：", layer2_output.shape)


#拆开第二残差阶段的第一个块，核对变换捷径与主分支相加
layer2_block=model.layer2[0]

with torch.no_grad():
    layer2_branch_output=layer2_block.conv1(layer1_output)
    layer2_branch_output=layer2_block.bn1(layer2_branch_output)
    layer2_branch_output = layer2_block.relu(layer2_branch_output)
    layer2_branch_output = layer2_block.conv2(layer2_branch_output)
    layer2_residual_output = layer2_block.bn2(layer2_branch_output)

    # projection 意为“投影”，这里用来表示捷径将输入变换到匹配的通道数和空间尺寸
    layer2_projection_output=layer2_block.downsample(layer1_output)

    layer2_sum_output=layer2_residual_output+layer2_projection_output

    layer2_manual_output = layer2_block.relu(layer2_sum_output.clone())

print("\n主分支输出形状：", layer2_residual_output.shape)
print("捷径分支输出形状：", layer2_projection_output.shape)
print("相加后的形状：", layer2_sum_output.shape)

print("主分支的部分数值：", layer2_residual_output[0, 0, 0, :5])
print("捷径分支的部分数值：", layer2_projection_output[0, 0, 0, :5])
print("相加后的部分数值：", layer2_sum_output[0, 0, 0, :5])
print("经过 ReLU 后：", layer2_manual_output[0, 0, 0, :5])

print(
    "第二阶段第一个块：拆开计算与直接调用是否一致：",
    torch.allclose(layer2_manual_output, layer2_block0_output),
)


#layer3 和 layer4。它们的结构规律与刚才的 layer2 相同：第一个块增加通道、缩小宽高，第二个块保持形状继续加工特征。
# 按顺序运行第三、第四残差阶段，观察各块的通道数和空间尺寸

with torch.no_grad():
    layer3_block0_output = model.layer3[0](layer2_output)
    layer3_output = model.layer3[1](layer3_block0_output)

    layer4_block0_output = model.layer4[0](layer3_output)
    layer4_output = model.layer4[1](layer4_block0_output)

print("\n第三阶段输入形状：", layer2_output.shape)
print("第三阶段第一个块输出形状：", layer3_block0_output.shape)
print("第三阶段第二个块输出形状：", layer3_output.shape)

print("\n第四阶段输入形状：", layer3_output.shape)
print("第四阶段第一个块输出形状：", layer4_block0_output.shape)
print("第四阶段第二个块输出形状：", layer4_output.shape)


#平均池化model.avgpool：把每张图片每个通道中的 7×7 共 49 个数，求平均，概括成一个数。这样每张图片就从 512 张小特征图，变成 512 个特征数，方便后续分类。
# model.avgpool打印出来的完整模块是AdaptiveAvgPool2d(output_size=(1, 1))，自适应（指定输出尺寸，让模块根据输入尺寸安排池化计算）平均池化

#观察全局平均池化，并核对第一张图片第一个通道的平均值
with torch.no_grad():
    avgpool_output=model.avgpool(layer4_output)

print("\n平均池化层：", model.avgpool)
print("平均池化输入形状：", layer4_output.shape)
print("平均池化输出形状：", avgpool_output.shape)

print("第一张图片第一个通道的 7×7 数值：")
print(layer4_output[0, 0])

print("手动求该通道的平均值：", layer4_output[0, 0].mean().item())
print("池化得到的对应数值：", avgpool_output[0, 0, 0, 0].item())


# 展平:每张图片的 512 个特征数整理成一行。分类层：计算 1000 个类别分数
# 最后，把逐层计算的结果与最开始的 model(images) 比较，验证整条数据流。

#模型里的分类层是：model.fc（fc是fully connected的缩写，意为“全连接”,这里的配置是：Linear(in_features=512, out_features=1000, bias=True)）


with torch.no_grad():
    flatten_output=torch.flatten(avgpool_output,start_dim=1)
    manual_logits=model.fc(flatten_output)

print("\n展平前形状：", avgpool_output.shape)
print("展平后形状：", flatten_output.shape)

print("\n分类层：", model.fc)
print("分类层权重形状：", model.fc.weight.shape)
print("分类层偏置形状：", model.fc.bias.shape)
print("最终分类分数形状：", manual_logits.shape)

print("逐层计算的前10个分数：", manual_logits[0, :10])
print("直接调用模型的前10个分数：", logits[0, :10])

print("完整前向结果是否一致：", torch.allclose(manual_logits, logits))
print("最大绝对差：", (manual_logits - logits).abs().max().item())