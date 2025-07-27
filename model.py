from torchvision import models
import torch
import torch.nn as nn

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

class ECABasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, k_size=3):
        super(ECABasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.eca = eca_layer(planes, k_size)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.eca(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def ResNet():

    model_ft = models.resnet101(pretrained=False)
    model_ft=nn.Sequential(*(list(model_ft.children())[:-2]))
    return model_ft



class IQC(nn.Module):
    def __init__(self,n_class):
        super(IQC, self).__init__()
        self.n_class=n_class

        self.Resnet1=ResNet()
        self.Resnet2=ResNet()

        self.SELayer=ECABasicBlock(inplanes=4096,planes=4096)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.conv1=nn.Sequential(nn.Conv2d(5,3,kernel_size=1, stride=1, bias=False))
        self.conv2=nn.Sequential(nn.Conv2d(5,3,kernel_size=1, stride=1, bias=False))

        self.fc1=nn.Sequential(nn.Linear(2048,  n_class),nn.Softmax(dim=1))
        self.fc2=nn.Sequential(nn.Linear(2048,  n_class),nn.Softmax(dim=1))
        self.fcStack=nn.Sequential(nn.Linear(2048,  n_class),nn.Softmax(dim=1))
        self.fcSum=nn.Sequential(nn.Linear(2048*2, 2048),nn.Linear(2048, n_class),nn.Softmax(dim=1))

    def forward(self,x,y):

        x=self.conv1(x)

        y=self.conv2(y)

        fc_res1 = self.Resnet1(x)
        fc_res2 = self.Resnet2(y)

        fc_resSum =torch.cat([fc_res2, fc_res1], dim=1)

        fc_resSum =self.SELayer(fc_resSum)

        fc_resSum = self.avgpool(fc_resSum)

        fc_resSum=fc_resSum.view(fc_resSum.size(0),-1)

        out4=self.fcSum(fc_resSum)

        return out4

