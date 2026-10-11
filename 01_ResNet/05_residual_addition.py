# 用小张量观察输入与残差分支逐元素相加，以及相加后的 ReLU 处理。

import torch 

block_input = torch.tensor([[[[3.0,-2.0],[1.0,4.0]]]])

residual_output=torch.tensor([[[[2.0,1.0],[-3.0,0.0]]]])


added_output=block_input+residual_output
block_output=torch.relu(added_output)

print("输入形状：", block_input.shape)
print("输入 x：", block_input)
print("残差分支 F(x)：", residual_output)
print("相加结果：", added_output)
print("经过 ReLU 的块输出：", block_output)

#让残差全部为零，观察输入是否被保留
# zeros_like 表示创建一个与指定张量形状、数据类型和设备相同，但数值全部为零的张量
zero_residual=torch.zeros_like(block_input)
zero_added_output=block_input+zero_residual
zero_block_output=torch.relu(zero_added_output)


print("零残差：", zero_residual)
print("零残差相加结果：", zero_added_output)
print("相加结果是否等于输入：",
      torch.equal(zero_added_output, block_input))
print("零残差经过 ReLU 的输出：", zero_block_output)
print("经过 ReLU 后是否等于输入：",
      torch.equal(zero_block_output, block_input))

#输入全部非负时，再观察零残差

nonnegative_input = torch.relu(block_input)
nonnegative_output = torch.relu(nonnegative_input + zero_residual)

print("非负输入：", nonnegative_input)
print("非负输入与零残差相加并经过 ReLU：", nonnegative_output)
print("最终输出是否等于非负输入：",
      torch.equal(nonnegative_output, nonnegative_input))