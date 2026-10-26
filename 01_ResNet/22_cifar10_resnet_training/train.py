#训练CIFAR ResNet，记录训练过程，保存模型并进行最终测试。

import argparse
from pathlib import Path
import time
import csv

import torch 
from torch import nn
from data import create_data_loaders
from model import CifarResNet

def get_learning_rate(completed_updates):
    if completed_updates<32000:
        return 0.1
    if completed_updates<48000:
        return 0.01
    return 0.001

def train(model,train_loader,device,log_path,total_updates=64000,print_every=100):
    if len(train_loader)==0:
        raise ValueError('训练加载器没有完整批次，请检查数据量和batch_size')

    with Path(log_path).open('x',encoding='utf-8',newline='')as file:
        writer=csv.writer(file)
        writer.writerow([
            'update',
            'learning_rate',
            'train_loss',   
            'train_accuracy',
            'lasting_seconds'
        ])
    
    criterion=nn.CrossEntropyLoss()
    optimizer=torch.optim.SGD(
        model.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=0.0001
    )
    completed_updates=0
    loss_sum=0.0
    correct_count=0
    sample_count=0
    started=time.perf_counter()
    model.train()

    while completed_updates<total_updates:
        for images,labels in train_loader:
            if completed_updates>=total_updates:
                break
            images = images.to(device)
            labels=labels.to(device)
            current_lr=get_learning_rate(completed_updates)
            for group in optimizer.param_groups:
                group['lr']=current_lr

            optimizer.zero_grad()

            outputs=model(images)
            loss=criterion(outputs,labels)
            if not torch.isfinite(loss).item():
                raise RuntimeError('训练损失不是有限数。')

            loss.backward()
            optimizer.step()

            completed_updates+=1
            loss_sum+=loss.item()*labels.size(0)
            correct_count+=(outputs.argmax(dim=1)==labels).sum().item()
            sample_count+=labels.size(0)

            if completed_updates%print_every==0 or completed_updates==total_updates:
                lasting_time=time.perf_counter()-started
                print(
                    f"更新={completed_updates}/{total_updates},学习率={current_lr},损失={loss_sum/sample_count:.4f},训练批次准确率={100*correct_count/sample_count:.2f}%,用时={lasting_time}s",flush=True
                )

                with Path(log_path).open('a',encoding='utf-8',newline='')as file:
                    writer=csv.writer(file)
                    writer.writerow([
                        completed_updates,
                        current_lr,
                        loss_sum/sample_count,
                        100*correct_count/sample_count,
                        lasting_time
                    ])

                loss_sum=0.0
                correct_count=0
                sample_count=0

    return time.perf_counter()-started

def evaluate(model,test_loader,device):
    criterion=nn.CrossEntropyLoss()
    loss_sum=0.0
    correct_count=0
    sample_count=0

    model.eval()
    with torch.no_grad():
        for images,labels in test_loader:
            images=images.to(device)
            labels=labels.to(device)

            outputs=model(images)
            loss=criterion(outputs,labels)
            loss_sum+=loss.item()*labels.size(0)
            correct_count+=(outputs.argmax(dim=1)==labels).sum().item()
            sample_count+=labels.size(0)

    return loss_sum/sample_count,100*correct_count/sample_count


def main():
    parser=argparse.ArgumentParser(description='实验22：CIFAR充分训练')
    parser.add_argument('--data-dir',required=True)
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--depth',type=int,choices=[20,56],default=56)
    args=parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError('本次训练需要使用CUDA')

    device=torch.device('cuda')
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    output_dir=Path(args.output_dir)
    output_dir.mkdir(parents=True,exist_ok=False)
    print('读取已有CIFAR-10，不重新下载。',flush=True)
    train_loader,test_loader=create_data_loaders(args.data_dir)
    model=CifarResNet(depth=args.depth).to(device)

    print(f"计算设备：{torch.cuda.get_device_name(0)},ResNet{args.depth},总批次128,正式训练64000次更新。",flush=True)
    training_seconds=train(model,train_loader,device,log_path=output_dir/'training.csv')

    model_path=output_dir/f'resnet{args.depth}.pt'
    torch.save(model.state_dict(),model_path)
    print('训练结束，最终权重已保存；',model_path,flush=True)

    test_loss,test_accuracy=evaluate(model,test_loader,device)

    result = (
        f"模型：CIFAR ResNet{args.depth}\n"
        "训练图片：50000；最终测试图片：10000\n"
        "批次大小：128；参数更新次数：64000；随机种子：42\n"
        "优化器：SGD；动量：0.9；权重衰减：0.0001\n"
        "学习率：0.1，完成32000次后0.01，完成48000次后0.001\n"
        f"训练耗时：{training_seconds / 60:.2f}分钟\n"
        f"最终测试损失：{test_loss:.4f}\n"
        f"最终测试准确率：{test_accuracy:.2f}%\n"
        f"最终测试错误率：{100 - test_accuracy:.2f}%\n"
    )

    (output_dir / "result.txt").write_text(result, encoding="utf-8")
    print(result)


if __name__ == "__main__":
    main()
    

