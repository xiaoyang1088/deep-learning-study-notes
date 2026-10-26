#检查自定义的ResNet50的结构、参考前向，以及一次梯度计算和数据更新

import argparse     #Python用来读取命令行参数的标准库。

import torch
from torch import nn
from torchvision.models import resnet50 as reference_resnet

from model import resnet50

#检查模型结构与参考实现
def check_model(device):
    torch.manual_seed(42)

    model=resnet50(num_classes=1000).to(device)
    reference_model=reference_resnet(weights=None).to(device)

    load_result=reference_model.load_state_dict(model.state_dict())

    model.eval()
    reference_model.eval()

    block_counts=[
        len(model.layer1),
        len(model.layer2),
        len(model.layer3),
        len(model.layer4)
    ]
    assert block_counts==[3,4,6,3]

    parameter_count=sum(
        parameter.numel()for parameter in model.parameters()
    )
    assert parameter_count==25557032

    print('计算设备：',device)
    print('阶段块数',block_counts)
    print('1000类模型参数量：',parameter_count)
    print('状态加载结果：',load_result)

    images=torch.randn(2,3,224,224,device=device)

    with torch.no_grad():
        output=model.conv1(images)
        output=model.bn1(output)
        output=model.relu(output)
        output=model.maxpool(output)

        assert tuple(output.shape)==(2,64,56,56)
        print('入口处理后：',output.shape)

        stages=[
            ('layer1',model.layer1,(2,256,56,56)),
            ("layer2", model.layer2, (2,512,28,28)),
            ("layer3", model.layer3, (2,1024,14,14)),
            ("layer4", model.layer4, (2,2048,7,7))
        ]

        for stage_name,stage,expected_shape in stages:
            output=stage(output)
            assert tuple(output.shape)==expected_shape
            print(f'{stage_name}',output.shape)

        output=model.avgpool(output)
        output=torch.flatten(output,1)
        manual_output=model.fc(output)

        direct_output=model(images)
        reference_output=reference_model(images)

    assert tuple(direct_output.shape)==(2,1000)
    torch.testing.assert_close(manual_output,direct_output,rtol=1e-4,atol=1e-5)
    torch.testing.assert_close(direct_output,reference_output,rtol=1e-4,atol=1e-5)

    print('与参考模型的最大绝对差：',
          (direct_output-reference_output).abs().max().item())
    print('1000类模型形状和参考前向检查通过。')

    del model,reference_model

    #实际检查使用的100类输出
    model100=resnet50(num_classes=100).to(device)
    model100.eval()

    with torch.no_grad():
        output100=model100(images)

    parameter_count100=sum(parameter.numel()for parameter in model100.parameters())

    assert tuple(output100.shape)==(2,100)
    assert parameter_count100==23712932

    #用随机小输入检查一次反向传播和参数更新
    model100.train()

    small_images=torch.randn(2,3,64,64,device=device)
    labels=torch.tensor([0,1],dtype=torch.long,device=device)

    criterion=nn.CrossEntropyLoss()
    optimizer=torch.optim.SGD(model100.parameters(),lr=0.01)

    weight_before=model100.fc.weight.detach().clone()   #detach表示不参与梯度追踪

    optimizer.zero_grad(0)
    logits=model100(small_images)
    loss=criterion(logits,labels)

    assert torch.isfinite(loss).item()      #检查是否为有限数值
    loss.backward()

    gradients_finite=all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all().item() for parameter in model100.parameters()
    )
    assert gradients_finite

    assert torch.equal(weight_before,model100.fc.weight)

    optimizer.step()

    weight_change=(model100.fc.weight.detach()-weight_before).abs().max().item()

    assert weight_change>0

    print('本次检查损失：',loss.item())
    print('所有可训练参数的梯度存在且有限',gradients_finite)
    print('分类层权重最大绝对变化：',weight_change)
    print('模型检查全部通过。')

#读取命令行参数并执行指定检查
def main():
    parser=argparse.ArgumentParser(description='检查自定义ResNet50模型')
    parser.add_argument('--check',choices=['model'],required=True)       #add_argument:添加一条参数规则
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')

    # 读取并解析命令行参数，将结果保存到args，供后续选择检查内容和计算设备
    args=parser.parse_args()    #parse_args:解析参数
    #args.check     得到字符串 "model"
    #args.device    得到字符串 "cuda"

    if args.device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('当前环境无法使用CUDA')
    device=torch.device(args.device)    #此时arg.device是一个类似于“CUDA”的字符串
    check_model(device)


if __name__=='__main__':
    main()