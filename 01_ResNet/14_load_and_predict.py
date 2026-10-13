#实验14：加载最佳模型，检查验证集表现，预测单张图片并保存正确和错误样本

from datetime import datetime
from pathlib import Path
import json
import runpy

import torch 
from torch import nn
from torch.utils.data import DataLoader,random_split
from torchvision import datasets,transforms
from torchvision.transforms.functional import to_pil_image

#读取实验13保存的最佳状态
base_dir=Path(__file__).resolve().parent
checkpoint_path=(
    base_dir/'outputs'/'experiment13_20261012_150229'/'best.pt'
)

checkpoint=torch.load(
    checkpoint_path,
    map_location='cpu',     #把保存的张量读到CPU
    weights_only=True       #使用受限加载方式，适合这里保存的张量和普通数据。这里仅加载自己生成的文件，不随意加载陌生来源文件。
)
config=checkpoint['config']
classes=checkpoint['classes']

device=torch.device('cuda'if torch.cuda.is_available()else 'cpu')

#创建相同结构并加载参数与BN运行统计量
experiment=runpy.run_path(str(base_dir/'08_build_resnet18.py'))
ResNet18=experiment['ResNet18']

model=ResNet18(num_classes=config['num_classes'])
model.conv1=nn.Conv2d(
    3,64,kernel_size=3,stride=1,padding=1,bias=False
)
model.maxpool=nn.Identity()

load_result=model.load_state_dict(checkpoint['model_state_dict'])
model=model.to(device)
model.eval()  

#重建原来的验证子集

full_train_dataset = datasets.CIFAR10(
    root=str(base_dir / "data"),
    train=True,
    transform=transforms.ToTensor(),
    download=False
)

train_dataset, val_dataset = random_split(
    full_train_dataset,
    [config["train_size"], config["val_size"]],
    generator=torch.Generator().manual_seed(config["seed"])
)

val_loader = DataLoader(
    val_dataset,
    batch_size=config["batch_size"],
    shuffle=False,
    num_workers=0
)

#在新目录中保存本次预测结果
run_name=datetime.now().strftime('experiment14_validation_%Y%m%d_%H%M%S')
output_dir=base_dir/'outputs'/run_name
output_dir.mkdir(parents=True,exist_ok=False)

print("加载文件：",checkpoint_path)
print("保存轮次:",checkpoint['epoch'])
print("状态加载结果：",load_result)
print("计算设备:",device)
print("验证样本数：",len(val_dataset))

#重新计算整个验证集指标
criterion=nn.CrossEntropyLoss()
loss_sum=0.0
correct_count=0
sample_count=0

with torch.no_grad():
    for images,labels in val_loader:
        images=images.to(device)
        labels=labels.to(device)

        logits=model(images)
        loss=criterion(logits,labels)

        batch_count=labels.size(0)
        loss_sum+=loss.item()*batch_count
        correct_count+=(logits.argmax(dim=1)==labels).sum().item()
        sample_count+=batch_count

val_loss=loss_sum/sample_count
val_accuracy=correct_count/sample_count
saved_row=checkpoint['history'][-1]

print(f'重新计算的验证损失：{val_loss:.4f}')
print(f'重新计算的验证准确率：{val_accuracy:.2%}')
print(f"原日志验证准确率：{saved_row['val_accuracy']:.2%}")

#单独预测验证子集的第一张图片
image,label = val_dataset[0]
single_input=image.unsqueeze(0).to(device)

with torch.no_grad():
    single_logits=model(single_input)
    single_probabilities=torch.softmax(single_logits,dim=1)

prediction=single_logits.argmax(dim=1).item()
confidence=single_probabilities[0,prediction].item()

to_pil_image(image).save(output_dir/'single_image.png')

print('这张图片的原始形状：',image.shape)
print('送入模型的形状：',single_input.shape)
print('单张分类分数形状：',single_logits.shape)
print('真实类别：',classes[label])
print('预测类别：',classes[prediction])
print(f'预测类别的模型概率：{confidence:.2%}')

#从验证集第一批中各选最多三张正确和错误样本
images,labels = next(iter(val_loader))

with torch.no_grad():
    logits=model(images.to(device))
    probabilities=torch.softmax(logits,dim=1).cpu()
    predictions=logits.argmax(dim=1).cpu()

saved_counts={
    'correct':0,
    "wrong":0
}
examples=[]

for sample_index in range(labels.size(0)):
    true_label=labels[sample_index].item()
    predicted_label=predictions[sample_index].item()

    category='correct'if true_label ==predicted_label else 'wrong'

    if saved_counts[category]>=3:
        continue

    saved_counts[category]+=1
    filename=(
        f'{category}_{sample_index:02d}_true_{classes[true_label]}_pred_{classes[predicted_label]}.png'
    )
    to_pil_image(images[sample_index]).save(output_dir/filename)

    record={
        'validation_index':sample_index,
        'correct':true_label==predicted_label,
        'true_class':classes[true_label],
        'predicted_class':classes[predicted_label],
        'predicted_probability':probabilities[sample_index,predicted_label].item(),
        'true_class_probability':probabilities[sample_index,true_label].item(),
        'image_file':filename
    }
    examples.append(record)

    print(
        f"样本={sample_index},真实={classes[true_label]},预测={classes[predicted_label]},预测类别概率={record['predicted_probability']:.2%},是否正确={record['correct']}"
    )

#保存指标及样本预测记录
summary = {
    "checkpoint": str(checkpoint_path),
    "checkpoint_epoch": checkpoint["epoch"],
    "dataset": "validation",
    "sample_count": sample_count,
    "loss": val_loss,
    "accuracy": val_accuracy,
    "saved_log_accuracy": saved_row["val_accuracy"],
    "single_prediction": {
        "true_class": classes[label],
        "predicted_class": classes[prediction],
        "predicted_probability": confidence,
        "image_file": "single_image.png"
    },
    "examples": examples
}

with (output_dir / "prediction_results.json").open(
    "w", encoding="utf-8"
) as result_file:
    json.dump(summary, result_file, ensure_ascii=False, indent=2)

print("保存的正确与错误样本数量：", saved_counts)
print("结果保存位置：", output_dir)