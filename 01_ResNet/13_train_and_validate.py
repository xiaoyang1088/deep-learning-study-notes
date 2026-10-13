# 实验13：完整训练和验证小图片版ResNet18，记录指标并保存模型状态
# 每轮遍历45000张训练图片，再检查5000张验证图片

# 运行后，会在该模型目录的 outputs 中创建一个带时间的文件夹,类似于experiment13_20261011_153000
# 文件夹里保存：
# config.json: configuration，配置；记录本次实验设置
# metrics.csv: metrics，指标；记录每轮损失与准确率
# best.pt: best，最佳；验证准确率最高时的状态
# last.pt: last，最后；最近一个完整训练轮次的状态

from datetime import datetime
from pathlib import Path
import csv
import json
import runpy
import time

import torch
from torch import nn 
from torch.utils.data import DataLoader,random_split
from torchvision import datasets,transforms

#设置实验配置并创建独立输出目录
config={
    "seed":42,
    "epochs":5,     #训练 5 轮
    "batch_size":64,
    "learning_rate":0.01,
    "optimizer":"SGD",
    "momentum":0.0,     #不使用动量
    "weight_decay":0.0,     #不使用权重衰减
    "train_size":45000,
    "val_size":5000,
    "num_classes":10,
    "model":"custom_resnet18_small_image",      #只是描述性的记录
    "transform":"ToTensor",
    "num_workers":0,
}

base_dir=Path(__file__).resolve().parent

run_name=datetime.now().strftime("experiment13_%Y%m%d_%H%M%S")
# datetime.now()：获取当前日期和时间
# .strftime(...)：把时间转换为指定格式的字符串(string format time)

output_dir=base_dir/"outputs"/run_name
output_dir.mkdir(parents=True,exist_ok=False)

#复用自定义模型，设置随机种子和设备

experiment=runpy.run_path(str(base_dir/"08_build_resnet18.py"))
ResNet18=experiment['ResNet18']

torch.manual_seed(config['seed'])
device=torch.device("cuda"if torch.cuda.is_available()else "cpu")

config['device']=str(device)
config["torch_version"]=str(torch.__version__)

#读取数据并沿用训练验证划分
full_train_dataset=datasets.CIFAR10(
    root=str(base_dir/"data"),
    train=True,
    transform=transforms.ToTensor(),
    download=False
)

train_dataset,val_dataset=random_split(
    full_train_dataset,
    [config['train_size'],config['val_size']],
    generator=torch.Generator().manual_seed(config['seed'])
)

train_loader=DataLoader(
    train_dataset,
    batch_size=config['batch_size'],
    shuffle=True,
    num_workers=config['num_workers'],
    generator=torch.Generator().manual_seed(config['seed'])
)

val_loader=DataLoader(
    val_dataset,
    batch_size=config['batch_size'],
    shuffle=False,
    num_workers=config['num_workers']
)

#创建模型、损失函数和优化器
model=ResNet18(num_classes=config['num_classes'])
model.conv1=nn.Conv2d(
    3,64,kernel_size=3,stride=1,padding=1,bias=False
)
model.maxpool=nn.Identity()
model=model.to(device)

criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(
    model.parameters(),
    lr=config['learning_rate'],
    momentum=config['momentum'],
    weight_decay=config['weight_decay']
)


#保存配置并建立指标表
with (output_dir/'config.json').open('w',encoding='utf-8') as config_file:
    json.dump(config,config_file,ensure_ascii=False,indent=2)     #config是要保存的数据，confige_file是写入文件，ensure_ascii=False表示不强制转化，中文可以保存为可读的中文，indent=2指每一层缩进2个空格，让文件易读，w清空或新建

fieldnames=[
    "epoch",'train_loss',"train_accuracy",
    "val_loss","val_accuracy","seconds"
]
metrics_path=output_dir/"metrics.csv"

with metrics_path.open('w',newline='',encoding='utf-8')as metrics_file:     #newline=""让 CSV 工具处理换行，避免出现多余空白行
    writer=csv.DictWriter(metrics_file,fieldnames=fieldnames)       #DictWriter是字典写入器，根据fieldnames确定列名和顺序
    writer.writeheader()    #写入表头。writer.writerow(row)：写一行具体数据

print("计算设备：", device)
print("训练样本数：", len(train_dataset))
print("验证样本数：", len(val_dataset))
print("每轮训练批次数：", len(train_loader))
print("输出目录：", output_dir)

#逐轮计算、验证并保存状态
best_val_accuracy=-1.0
history=[]

for epoch in range(1,config['epochs']+1):
    start_time=time.perf_counter()      #performance counter，意思是性能计时器
    model.train()

    train_loss_sum=0.0
    train_correct=0
    train_count=0

    for batch_index,(images,labels) in enumerate(train_loader,start=1):     #enumerate加编号
        images=images.to(device)
        labels=labels.to(device)

        optimizer.zero_grad(0)
        logits=model(images)
        loss=criterion(logits,labels)
        loss.backward()
        optimizer.step()

        batch_count=labels.size(0)      #0代表0维里的数量
        train_loss_sum+=loss.item()*batch_count
        train_correct+=(logits.argmax(dim=1)==labels).sum().item()
        train_count+=batch_count

        if batch_index%100==0 or batch_index==len(train_loader):
            print(
                f"轮次={epoch}，批次={batch_index}/{len(train_loader)}，"
                f"累计训练损失={train_loss_sum / train_count:.4f}"
            )

    train_loss=train_loss_sum/train_count
    train_accuracy=train_correct/train_count

    model.eval()
    val_loss_sum=0.0
    val_correct=0
    val_count=0

    with torch.no_grad():
        for images,labels in val_loader:                
            images=images.to(device)
            labels=labels.to(device)

            logits=model(images)
            loss=criterion(logits,labels)

            batch_count=labels.size(0)
            val_loss_sum+=loss.item()*batch_count
            val_correct+=(logits.argmax(dim=1)==labels).sum().item()
            val_count+=batch_count

    val_loss=val_loss_sum/val_count
    val_accuracy=val_correct/val_count
    seconds=time.perf_counter()-start_time

    row={
        "epoch":epoch,
        'train_loss':train_loss,
        "train_accuracy":train_accuracy,
        "val_loss":val_loss,
        "val_accuracy":val_accuracy,
        'seconds':seconds
    }
    history.append(row)

    with metrics_path.open('a',newline='',encoding='utf-8') as metrics_file:
        writer=csv.DictWriter(metrics_file,fieldnames=fieldnames)
        writer.writerow(row)

    is_best=val_accuracy>best_val_accuracy
    if is_best:
        best_val_accuracy=val_accuracy

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_accuracy": best_val_accuracy,
        "config": config,
        "classes": full_train_dataset.classes,
        "history": history
    }

    torch.save(checkpoint,output_dir/'last.pt')
    if is_best:
        torch.save(checkpoint,output_dir/'best.pt')
    print(
        f"轮次={epoch}/{config['epochs']}，"
        f"训练损失={train_loss:.4f}，训练准确率={train_accuracy:.2%}，"
        f"验证损失={val_loss:.4f}，验证准确率={val_accuracy:.2%}，"
        f"训练与验证耗时={seconds:.1f}秒，保存最佳={is_best}"
    )
    
    
print("训练结束，最佳验证准确率：", f"{best_val_accuracy:.2%}")
print("结果保存位置：", output_dir)