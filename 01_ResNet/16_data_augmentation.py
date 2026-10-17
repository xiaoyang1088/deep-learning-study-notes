#实验16：数据增强实验，保持相同初始状态和数据划分，对比有无随机裁剪、水平翻转的训练效果

from datetime import datetime
from pathlib import Path
import csv
import json
import runpy
import time

import torch 
from torch import nn
from torch.utils.data import DataLoader,random_split,Subset
from torchvision import transforms,datasets
from torchvision.transforms.functional import to_pil_image

#设置共同配置并读取模型定义
base_dir=Path(__file__).resolve().parent

config={
    'seed':42,
    'epochs':5,
    'batch_size':64,
    'learning_rate':0.01,
    'optimizer':'SGD',
    'momentum':0.0,
    'weight_decay':0.0,
    'train_size':45000,
    'val_size':5000,
    'num_classes':10,
    'model':'custom_resnet18_small_image',
    'num_workers':0,
    'bn_calibration':False,
    'training_transforms':{
        'baseline':"ToTensor",
        'augmented':"RandomCrop(32,padding=4)+RandomHorizontalFlip(p=0.5)+ToTensor"
    },
    'validation_transform':"ToTensor"
}

experiment=runpy.run_path(str(base_dir/'08_build_resnet18.py'))
ResNet18=experiment['ResNet18']

device=torch.device('cuda'if torch.cuda.is_available()else 'cpu')
config['device']=str(device)
config['torch_version']=str(torch.__version__)

def create_model():
    model=ResNet18(num_classes=config['num_classes'])
    model.conv1=nn.Conv2d(
        3,64,kernel_size=3,stride=1,padding=1,bias=False
    )
    model.maxpool=nn.Identity()
    return model



#固定训练、验证图片和编号
plain_dataset=datasets.CIFAR10(
    root=str(base_dir/'data'),
    train=True,     #指官方训练集的50000张图片
    transform=transforms.ToTensor(),
    download=False
)

train_subset,val_dataset =random_split(
    plain_dataset,
    [config['train_size'],config['val_size']],
    generator=torch.Generator().manual_seed(config['seed'])
)

train_indices=list(train_subset.indices)

augmentation_transform=transforms.Compose([
    transforms.RandomCrop(32,padding=4),    #把原来32*32的图像四周填充4个像素，得到40*40的图像，默认填充值为0，然后截取一个32*32的区域
    transforms.RandomHorizontalFlip(p=0.5),     #Horizontal代表水平方向，flip是翻转，代码意思为它有50%的概率左右翻转。
    transforms.ToTensor()
])

augmented_dataset=datasets.CIFAR10(
    root=str(base_dir/'data'),
    train=True,
    transform=augmentation_transform,
    download=False
)

training_datasets={
    'baseline':Subset(plain_dataset,train_indices),
    'augmented':Subset(augmented_dataset,train_indices)
}


#创建输出目录并保存增强预览
run_name=datetime.now().strftime('experiment16_%Y%m%d_%H%M%S')
output_dir=base_dir/'outputs'/run_name
output_dir.mkdir(parents=True,exist_ok=False)

preview_dir=output_dir/'preview'
preview_dir.mkdir()

sample_index=train_indices[0]
original_image,original_label=plain_dataset[sample_index]
to_pil_image(original_image).save(preview_dir/'original.png')

torch.manual_seed(123)
for preview_index in range(3):
    augmented_image,augmented_label=augmented_dataset[sample_index]
    assert  augmented_label==original_label
    to_pil_image(augmented_image).save(preview_dir/f'augmented_{preview_index:02d}.png')

print('预览图片类别：',plain_dataset.classes[original_label])
print('预览图片目录：',preview_dir)

#保存两组共同的初始状态
torch.manual_seed(config['seed'])
initial_model=create_model()

initial_state={}
for name,tensor in initial_model.state_dict().items():
    initial_state[name]=tensor.detach().clone()
del initial_model

torch.save(initial_state,output_dir/'initial_state.pt')

with(output_dir/'config.json').open('w',encoding='utf-8')as config_file:
    json.dump(config,config_file,ensure_ascii=False,indent=2)

metric_fields=[
    'epoch',
    'train_loss',
    'train_accuracy',
    'val_loss',
    'val_accuracy',
    'seconds'
]

summary_fields=[
    'group',
    'best_epoch',
    'best_val_accuracy',
    'final_train_accuracy',
    'final_val_accuracy'
]

summary_rows=[]

def train_one_epoch(model,loader,criterion,optimizer):
    model.train()
    loss_sum=0.0
    correct_count=0
    sample_count=0

    for images,labels in loader:
        images=images.to(device)
        labels=labels.to(device)

        optimizer.zero_grad(0)
        logits=model(images)
        loss=criterion(logits,labels)

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

            batch_count=labels.size(0)
            loss_sum+=loss.item()*batch_count
            correct_count+=(logits.argmax(dim=1)==labels).sum().item()
            sample_count+=batch_count

    return loss_sum/sample_count,correct_count/sample_count

print("计算设备",device)
print("训练与验证样本数：",len(train_indices),len(val_dataset))
print("输出目录：",output_dir)

#两组分别恢复起点、训练并验证
for group_name,train_dataset in training_datasets.items():
    torch.manual_seed(config['seed'])

    train_loader=DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        generator=torch.Generator().manual_seed(config['seed'])
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        generator=torch.Generator().manual_seed(config['seed'])
    )

    model=create_model()
    model.load_state_dict(initial_state)

    same_initial_state=all(
        torch.equal(tensor,initial_state[name]) for name ,tensor in model.state_dict().items()
    )   #all表示内部必须全部为true，才为true
    assert same_initial_state

    model=model.to(device)

    criterion=nn.CrossEntropyLoss()
    optimizer=torch.optim.SGD(
        model.parameters(),
        lr=config['learning_rate'],
        momentum=config['momentum'],
        weight_decay=config['weight_decay']
    )

    trial_dir=output_dir/group_name
    trial_dir.mkdir()

    trial_config=dict(config)
    trial_config['group']=group_name
    trial_config['transform']=config['training_transforms'][group_name]

    with (trial_dir/'config.json').open('w',encoding='utf-8')as config_files:
        json.dump(trial_config,config_files,ensure_ascii=False,indent=2)

    metrics_path=trial_dir/'metrics.csv'
    with metrics_path.open('w',newline='',encoding='utf-8')as metrics_file:
        writer=csv.DictWriter(metrics_file,fieldnames=metric_fields)
        writer.writeheader()

    best_val_accuracy=-1.0
    best_epoch=0
    history=[]

    print(f'\n开始组别={group_name},初始状态一致={same_initial_state}')
    for epoch in range(1,config['epochs']+1):
        start_time=time.perf_counter()

        train_loss,train_accuracy=train_one_epoch(model,train_loader,criterion,optimizer)
        val_loss,val_accuracy=evaluate(model,val_loader,criterion)

        seconds=time.perf_counter()-start_time

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "seconds": seconds,
        }
        history.append(row)

        with metrics_path.open("a", newline="", encoding="utf-8") as metrics_file:
            writer = csv.DictWriter(metrics_file, fieldnames=metric_fields)
            writer.writerow(row)

        is_best = val_accuracy > best_val_accuracy
        if is_best:
            best_val_accuracy = val_accuracy
            best_epoch = epoch

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_accuracy": best_val_accuracy,
            "config": trial_config,
            "classes": plain_dataset.classes,
            "history": history,
        }

        torch.save(checkpoint, trial_dir / "last.pt")
        if is_best:
            torch.save(checkpoint, trial_dir / "best.pt")

        print(
            f"组别={group_name}，轮次={epoch}/{config['epochs']}，"
            f"训练损失={train_loss:.4f}，训练准确率={train_accuracy:.2%}，"
            f"验证损失={val_loss:.4f}，验证准确率={val_accuracy:.2%}，"
            f"耗时={seconds:.1f}秒，保存最佳={is_best}"
        )

    summary_rows.append({
        "group": group_name,
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_accuracy,
        "final_train_accuracy": train_accuracy,
        "final_val_accuracy": val_accuracy,
    })

    with (output_dir / "summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    del checkpoint, optimizer, model

print("\n数据增强对照结果：")
for row in summary_rows:
    print(
        f"组别={row['group']}，最佳轮次={row['best_epoch']}，"
        f"最佳验证准确率={row['best_val_accuracy']:.2%}，"
        f"最后验证准确率={row['final_val_accuracy']:.2%}"
    )

print("结果保存位置：", output_dir)