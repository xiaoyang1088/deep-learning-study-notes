import argparse
import json
from pathlib import Path
import time 
import csv

import torch
from torch import nn 
from torch.utils.data import DataLoader

from data import ManifestDataset,build_transform,create_data_loaders

from model import resnet50

def get_learning_rate(epoch):
    if epoch<=30:
        return 0.1
    if epoch<=60:
        return 0.01
    if epoch <=90:
        return 0.001
    return 0.0001

def train_one_epoch(model,loader,criterion,optimizer,device):
    model.train()
    loss_sum=0.0
    correct_count=0
    sample_count=0

    for batch_index,(images,labels)in enumerate(loader,start=1):
        images=images.to(device)
        labels=labels.to(device)
        optimizer.zero_grad()
        outputs=model(images)
        loss=criterion(outputs,labels)
        if not torch.isfinite(loss).item():
            raise RuntimeError('训练损失不是有限数。')
        loss.backward()
        optimizer.step()
        loss_sum+=loss.item()*labels.size(0)
        correct_count+=(outputs.argmax(dim=1)==labels).sum().item()
        sample_count+=labels.size(0)

        if batch_index%100==0 or batch_index==len(loader):
            print(f"训练批次={batch_index}/{len(loader)},损失={loss_sum/sample_count:.4f},准确率={100*correct_count/sample_count:.2f}%",flush=True)

    if sample_count==0:
        raise ValueError('没有完整训练批次。')

    return loss_sum/sample_count,100*correct_count/sample_count

def evaluate(model,loader,criterion,device):
    loss_sum=0.0
    correct_count=0
    sample_count=0
    model.eval()
    with torch.no_grad():
        for images,labels in loader:
            images=images.to(device)
            labels=labels.to(device)

            outputs=model(images)
            loss=criterion(outputs,labels)
            loss_sum+=loss.item()*labels.size(0)
            correct_count+=(outputs.argmax(dim=1)==labels).sum().item()
            sample_count+=labels.size(0)

    return loss_sum/sample_count,100*correct_count/sample_count

def main():
    parser=argparse.ArgumentParser(description='实验21：ResNet50训练、验证和最终测试')
    parser.add_argument('--project-dir',required=True)
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--epochs',type=int,default=100)
    parser.add_argument('--batch-size',type=int,default=128)
    parser.add_argument('--num-workers',type=int,default=4)
    args=parser.parse_args()

    if args.epochs<1 or args.batch_size<2 or args.num_workers<0:
        parser.error('epochs至少1，batch_size至少2，num_workers不能为负数。')
    if not torch.cuda.is_available():
        raise RuntimeError('本次训练需要使用CUDA，请先检查环境。')
    project_dir=Path(args.project_dir).resolve()
    data_root=project_dir/'data/imagenet100_raw'
    manifest_dir=project_dir/'metadata/splits_seed42'
    classes_path=project_dir/'metadata/classes.json'
    audit=json.loads((manifest_dir/'data_audit.json').read_text(encoding='utf-8'))
    if audit['status']!='complete':
        raise RuntimeError('请先完成图片检查、清理与固定划分。')
    output_dir=Path(args.output_dir)
    output_dir.mkdir(parents=True,exist_ok=False)
    device=torch.device('cuda')
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    print('读取训练与验证清单。',flush=True)
    train_loader,val_loader=create_data_loaders(
        data_root,
        manifest_dir,
        classes_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    classes=train_loader.dataset.classes
    model=resnet50(num_classes=len(classes)).to(device)
    criterion=nn.CrossEntropyLoss()
    optimizer=torch.optim.SGD(
        model.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=0.0001
    )
    #创建每轮训练和验证结果表
    log_path=output_dir/'training.csv'
    with log_path.open('x',encoding='utf-8',newline='')as file:
        writer=csv.writer(file)
        writer.writerow([
            'epoch',
            'learning_rate',
            'train_loss',
            'train_accuracy',
            'val_loss',
            'val_accuracy',
            'epoch_seconds',
            'elapsed_seconds'
        ])

    best_accuracy=-1.0
    best_epoch=0
    started=time.perf_counter()
    print(
        f"计算设备：{torch.cuda.get_device_name(0)}\n类别数：{len(classes)}\n训练图片：{len(train_loader.dataset)}\n验证图片:{len(val_loader.dataset)}",flush=True
    )
    for epoch in range(1,args.epochs+1):
        epoch_started=time.perf_counter()
        current_lr=get_learning_rate(epoch=epoch)
        for group in optimizer.param_groups:
            group['lr']=current_lr
        print(f"轮次={epoch}/{args.epochs},学习率={current_lr}",flush=True)
        train_loss,train_accuracy=train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )
        val_loss,val_accuracy=evaluate(model,val_loader,criterion,device)

        checkpoint={
            'model_state_dict':model.state_dict(),
            'classes':classes,
            'epoch':epoch,
            'val_accuracy':val_accuracy,
            'settings':vars(args)   #vars(args)转换为字典
        }

        if val_accuracy>best_accuracy:
            best_accuracy=val_accuracy
            best_epoch=epoch
            torch.save(checkpoint,output_dir/'best.pt')
        epoch_seconds=time.perf_counter()-epoch_started
        elapsed_seconds=time.perf_counter()-started
        with log_path.open('a',encoding='utf-8',newline='')as file:
            writer=csv.writer(file)
            writer.writerow([
                epoch,
                current_lr,
                train_loss,
                train_accuracy,
                val_loss,
                val_accuracy,
                epoch_seconds,
                elapsed_seconds
            ])
        print(
            f"训练损失={train_loss:.4f},训练准确率={train_accuracy:.2f}%,验证损失={val_loss:.4f},验证准确率={val_accuracy:.2f}%,本轮用时={epoch_seconds}s",
            flush=True
        )

    torch.save(checkpoint,output_dir/'final.pt')
    training_seconds=time.perf_counter()-started

    best_checkpoint=torch.load(
        output_dir/'best.pt',
        map_location='cpu'
    )
    model.load_state_dict(best_checkpoint['model_state_dict'])
    test_dataset=ManifestDataset(
        data_root,
        manifest_dir/'test.csv',
        classes_path,
        build_transform(False)
    )
    test_loader=DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False
    )
    test_loss,test_accuracy=evaluate(
        model,
        test_loader,
        criterion,
        device
    )
    result=(
        f"模型：自定义ResNet50；类别数：{len(classes)}\n"
        f"计算设备：{torch.cuda.get_device_name(0)}\n"
        f'训练图片：{len(train_loader.dataset)}\n'
        f'验证图片：{len(val_loader.dataset)}\n'
        f'测试图片：{len(test_loader.dataset)}\n'
        f"训练轮数：{args.epochs}；批次大小：{args.batch_size}；随机种子：42\n"
        "优化器：SGD；动量：0.9；权重衰减：0.0001\n"
        "学习率：1至30轮0.1，31至60轮0.01，61至90轮0.001，91轮起0.0001\n"
        f"最佳验证轮次：{best_epoch}；验证准确率：{best_accuracy:.2f}%\n"
        f"最佳验证模型的最终测试损失：{test_loss:.4f}\n"
        f"最终测试准确率：{test_accuracy:.2f}%；错误率：{100 - test_accuracy:.2f}%\n"
        f'总耗时：{training_seconds}s\n'
    )

    (output_dir/'result.txt').write_text(result,encoding='utf-8')
    print(result,flush=True)

if __name__ == '__main__':
    main()
