from torch import nn
from torch.nn import functional 

class OptionAShortcut(nn.Module):
    def forward(self,inputs):
        outputs=inputs[:,:,::2,::2]
        padding=inputs.shape[1]//2

        return functional.pad(
            outputs,
            (0,0,0,0,padding,padding)
        )
        #| 数字位置 |   控制内容   |
        #| 第1、2个 | 宽度左右两侧 |
        #| 第3、4个 | 高度上下两侧 |
        #| 第5、6个 | 通道前后两侧 |

class BasciBlock(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1):
        super().__init__()
        self.conv1=nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn1=nn.BatchNorm2d(out_channels)
        self.conv2=nn.Conv2d(out_channels,out_channels,3,1,1,bias=False)
        self.bn2=nn.BatchNorm2d(out_channels)
        self.shortcut=nn.Identity()if stride==1 else OptionAShortcut()

    def forward(self,inputs):
        outputs=self.conv1(inputs)
        outputs=self.bn1(outputs)
        outputs=functional.relu(outputs)
        outputs=self.conv2(outputs)
        outputs=self.bn2(outputs)
        outputs=outputs+self.shortcut(inputs)
        outputs=functional.relu(outputs)
        return outputs

class CifarResNet(nn.Module):
    def __init__(self,depth=56):
        super().__init__()
        if depth not in {20,56}:
            raise ValueError("本实验采用20层或56层。")
        block_per_stage=(depth-2)//6
        self.conv1=nn.Conv2d(3,16,3,1,1,bias=False)
        self.bn1=nn.BatchNorm2d(16)
        self.layer1=self.make_layer(16,16,block_per_stage,1)
        self.layer2=self.make_layer(16,32,block_per_stage,2)
        self.layer3=self.make_layer(32,64,block_per_stage,2)
        self.fc=nn.Linear(64,10)

        for module in self.modules():
            if isinstance(module,nn.Conv2d):
                nn.init.kaiming_normal_(module.weight,mode= 'fan_in',nonlinearity='relu')
            elif isinstance(module,nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module,nn.Linear):
                nn.init.normal_(module.weight,mean=0.0,std=0.01)
                nn.init.zeros_(module.bias )


    def make_layer(self,in_channels,out_channels,count,stride):
        blocks=[BasciBlock(in_channels,out_channels,stride)] 
        for block_index in range(1,count):
            blocks.append(BasciBlock(out_channels,out_channels))
        return nn.Sequential(*blocks)

    def forward(self,inputs):
        outputs=self.conv1(inputs)
        outputs=self.bn1(outputs)
        outputs=functional.relu(outputs)
        outputs=self.layer1(outputs)
        outputs=self.layer2(outputs)
        outputs=self.layer3(outputs)
        outputs=outputs.mean(dim=(2,3))
        outputs=self.fc(outputs)
        return outputs
            
