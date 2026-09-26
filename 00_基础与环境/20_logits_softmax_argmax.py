#学习分类任务中的原始分数、softmax概率和argmax类别预测
#使用两组模拟输出，观察每个样本的类别概率和预测结果

import torch   

class_names=["猫","狗","鸟"]

logits=torch.tensor([
    [2.0,1.0,0.0],
    [0.0,1.0,3.0]
])

probabilities=torch.softmax(logits,dim=1)
probability_sums=probabilities.sum(dim=1)

predicted_classes=torch.argmax(logits,dim=1)
classes_from_probabilities=torch.argmax(probabilities,dim=1)

print("原始分类分数：")
print(logits)
print("分数形状：", logits.shape)

print("\n类别概率：")
print(probabilities)
print("概率形状：", probabilities.shape)

print("\n每个样本的概率之和：")
print(probability_sums)

print("\n直接根据分数预测的类别编号：")
print(predicted_classes)

print("\n根据概率预测的类别编号：")
print(classes_from_probabilities)

print("\n预测类别名称：")
for sample_index,class_index in enumerate(predicted_classes.tolist()):
    print("样本编号：",sample_index,"预测类别：",class_names[class_index])