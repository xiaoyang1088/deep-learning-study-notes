#实验15：使用相同初始状态和数据顺序，对比三种学习率的训练和验证表现

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

#设置共同配置并读取模型定义
base_dir = Path(__file__).resolve().parent

config={
    "seed":42,
    "epochs":5,
    "batch_size":64,
    "learning_rates":[0.001,0.01,0.1],
    'optimizer':'SGD',
    'momentum':0.0,
    'weight_decay':0.0,
    'train_size':45000,
    'val_size':5000,
    "num_classes":10,
    'model':"custom_resnet18_small_image",
    "transform":"ToTensor",
    "num_workers":0,
    "bn_calibration":False   
}

experiment=runpy.run_path(str(base_dir/'08_build_resnet18.py'))
ResNet18=experiment['ResNet18']

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config['device']=str(device)
config['torch_version']=str(torch.__version__)


def create_model():
    model=ResNet18(num_classes=config['num_classes'])
    model.conv1=nn.Conv2d(
        3,64,kernel_size=3,stride=1,padding=1,bias=False
    )
    model.maxpool=nn.Identity()
    return model

#固定数据划分和模型初始状态
full_train_dataset=datasets.CIFAR10(
    root=str(base_dir/'data'),
    train=True,
    transform=transforms.ToTensor(),
    download=False
)

train_dataset,val_dataset=random_split(
    full_train_dataset,
    [config['train_size'],config['val_size']],
    generator=torch.Generator().manual_seed(config['seed'])
)

torch.manual_seed(config['seed'])
initial_model=create_model()
initial_state={}
for name,tensor in initial_model.state_dict().items():
    initial_state[name]=tensor.detach().clone()
#initial_model.state_dict()取得模型状态字典，包括权重、偏置和BN运行统计量等
#initial_state 就是一份独立的初始存档，张量与模型内部共享的存储分开防止同时变化

del initial_model


#创建本次对照实验的输出目录
run_name=datetime.now().strftime('experiment15_%Y%m%d_%H%M%S')
output_dir=base_dir/'outputs'/run_name
output_dir.mkdir(parents=True,exist_ok=False)

with(output_dir/'config.json').open('w',encoding='utf-8')as config_file:
    json.dump(config,config_file,ensure_ascii=False,indent=2)   #dump把数据转换为JSON
torch.save(initial_state,output_dir/'initial_state.pt')

metric_fields=[
    'epoch',
    'train_loss',
    'train_accuracy',
    'val_loss',
    'val_accuracy',
    'seconds'
]

summary_fields=[
    'learning_rate',
    'best_epoch',
    'best_val_accuracy',
    'final_train_accuracy',
    'final_val_accuracy'
]

summary_rows=[]

def train_one_epoch(model,loader,criterion,optimizer):
    model.train()
    loss_sum=0.0
    correct_count=0.0
    sample_count=0

    for images,labels in loader:
        images=images.to(device)
        labels=labels.to(device)

        optimizer.zero_grad(0)
        logits=model(images)
        loss=criterion(logits,labels)

        if not torch.isfinite(loss).item():
            raise RuntimeError('训练损失出现NaN或Inf,请保留输出检查')#finite意思为有限的，NaN是not a number,inf为无穷

        
        loss.backward()
        optimizer.step()
        batch_count=labels.size(0)
        loss_sum+=loss.item()*batch_count
        correct_count+=(logits.argmax(dim=1)==labels).sum().item()
        sample_count+=batch_count
    return loss_sum/sample_count,correct_count/sample_count

def evaluate(model,loader,criterion):
    model.eval()
    loss_sum=0.0
    correct_count=0
    sample_count=0

    with torch.no_grad():
        for images,labels in loader:
            images=images.to(device)
            labels=labels.to(device)

            logits=model(images)
            loss=criterion(logits,labels)

            if not torch.isfinite(loss).item():
                raise RuntimeError("验证损失出现NaN或Inf,请保留输出并检查")

            batch_count=labels.size(0)
            loss_sum+=loss.item()*batch_count
            correct_count+=(logits.argmax(dim=1)==labels).sum().item()
            sample_count+=batch_count

        return loss_sum/sample_count,correct_count/sample_count

print("计算设备：",device)
print('训练与验证样本数：',len(train_dataset),len(val_dataset))
print('待比较学习率：',config['learning_rates'])
print('输出目录：',output_dir)

#每组重新恢复初始状态和数据顺序，再独立训练
for learning_rate in config['learning_rates']:
    torch.manual_seed(config['seed'])

    train_loader=DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,   #true代表打乱顺序
        num_workers=config['num_workers'],
        generator=torch.Generator().manual_seed(config['seed'])
    )    

    val_loader=DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers']
    )

    model=create_model()
    model.load_state_dict(initial_state)

    same_initial_state=all(
        torch.equal(tensor,initial_state[name]) for name,tensor in model.state_dict().items() 
    )
    assert same_initial_state
    model=model.to(device)

    criterion=nn.CrossEntropyLoss()
    optimizer=torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=config['momentum'],
        weight_decay=config['weight_decay']
    )

    trial_dir=output_dir/f'lr_{learning_rate:g}'    #:g通用数值格式
    trial_dir.mkdir()

    trial_config=dict(config)   #浅复制一个config字典
    trial_config['learning_rate']=learning_rate

    with (trial_dir/'config.json').open('w',encoding='utf-8')as config_file:
        json.dump(trial_config,config_file,ensure_ascii=False,indent=2)

    metrics_path=trial_dir/'metrics.csv'
    with metrics_path.open('w',newline='',encoding='utf-8')as metrics_file:
        writer=csv.DictWriter(metrics_file,fieldnames=metric_fields)
        writer.writeheader()

    best_val_accuracy=-1.0
    best_epoch=0
    history=[]

    print(f"\n开始学习率={learning_rate},初始状态一致={same_initial_state}")

    for epoch in range(1,config['epochs']+1):
        start_time=time.perf_counter()

        train_loss,train_accuracy = train_one_epoch(model,train_loader,criterion,optimizer)

        val_loss,val_accuracy=evaluate(model,val_loader,criterion)

        seconds=time.perf_counter()-start_time

        row={
            'epoch':epoch,
            'train_loss':train_loss,
            'train_accuracy':train_accuracy,
            'val_loss':val_loss,
            'val_accuracy':val_accuracy,
            'seconds':seconds
        }
        history.append(row)

        with metrics_path.open('a',newline='',encoding='utf-8')as metrics_file:
            writer=csv.DictWriter(metrics_file,fieldnames=metric_fields)
            writer.writerow(row)

        is_best=val_accuracy>best_val_accuracy
        if is_best:
            best_val_accuracy=val_accuracy
            best_epoch=epoch

        checkpoint={
            'epoch':epoch,
            'model_state_dict':model.state_dict(),
            'optimizer_state_dict':optimizer.state_dict(),
            'best_val_accuracy':best_val_accuracy,
            'config':trial_config,
            'classes':full_train_dataset.classes,
            'history':history
        }    

        torch.save(checkpoint,trial_dir/'last.pt')
        if is_best:
            torch.save(checkpoint,trial_dir/'best.pt')
        print(
            f"学习率={learning_rate}，轮次={epoch}/{config['epochs']}，"
            f"训练损失={train_loss:.4f}，训练准确率={train_accuracy:.2%}，"
            f"验证损失={val_loss:.4f}，验证准确率={val_accuracy:.2%}，"
            f"耗时={seconds:.1f}秒，保存最佳={is_best}"
        )

    summary_rows.append({
        'learning_rate':learning_rate,            'best_epoch':best_epoch,
        'best_val_accuracy':best_val_accuracy,
        'final_train_accuracy':train_accuracy,
        'final_val_accuracy':val_accuracy
        })

    with (output_dir / "summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    del checkpoint, optimizer, model

print("\n学习率对照结果：")
for row in summary_rows:
    print(
        f"学习率={row['learning_rate']}，"
        f"最佳轮次={row['best_epoch']}，"
        f"最佳验证准确率={row['best_val_accuracy']:.2%}，"
        f"最后验证准确率={row['final_val_accuracy']:.2%}"
    )

print("结果保存位置：", output_dir)