# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   1378453948@qq.com
@DateTime :   2023/11/07 13:46
@Version  :   1.0
@License  :   (C)Copyright 2023
"""
import torch.nn as nn
import torchvision.models as models
from collections import OrderedDict
from collections import namedtuple

res = {
    'resnet101': models.resnet101,
    'resnet50': models.resnet50,
    'resnet18': models.resnet18,
    'resnet34': models.resnet34,
    'resnet152': models.resnet152,
}


class ResNet(nn.Module):
    def __init__(self, backbone, pretrained=True):
        super(ResNet, self).__init__()
        resnet = res[backbone](pretrained=pretrained)  # pretrained ImageNet
        self.topconvs = nn.Sequential(
            OrderedDict(list(resnet.named_children())[0:3]))
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x):
        x = self.topconvs(x)
        layer0 = x
        x = self.max_pool(x)
        x = self.layer1(x)
        layer1 = x
        x = self.layer2(x)
        layer2 = x
        x = self.layer3(x)
        layer3 = x
        x = self.layer4(x)
        layer4 = x
        res_outputs = namedtuple("SideOutputs", ['layer0', 'layer1', 'layer2', 'layer3', 'layer4'])
        out = res_outputs(layer0=layer0, layer1=layer1, layer2=layer2, layer3=layer3, layer4=layer4)
        return out


