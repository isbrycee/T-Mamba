# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   1378453948@qq.com
@DateTime :   2023/11/07 13:46
@Version  :   1.0
@License  :   (C)Copyright 2023
"""
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch
from collections import namedtuple

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


# from torchvision.models import vgg16_bn
class VGG(nn.Module):
    def __init__(self, features, backbone='vgg16_bn', cfg_key='D', pretrained=False,
                 batch_norm=False):
        super(VGG, self).__init__()
        # l = len(cfg[cfg_key])
        # zip_list = zip(*(range(l), cfg[cfg_key]))
        # id1 = [z[0] for i, z in enumerate(zip_list) if z[1] == 'M']
        # id1= [z * 3 if batch_norm else z * 2 for i,z in enumerate(id1)]
        if pretrained:
            feature_state = features.state_dict()
            vgg_dict = model_zoo.load_url(model_urls[backbone])
            pretrain_dict = {k[9:]: v for k, v in vgg_dict.items() if k[9:] in feature_state.keys()}
            feature_state.update(pretrain_dict)
            features.load_state_dict(feature_state)
        id1 = []
        count = 0
        increase = 3 if batch_norm else 2
        for ii in range(len(cfg[cfg_key])):
            count = count + increase if cfg[cfg_key][ii] != 'M' else count + 1
            if cfg[cfg_key][ii] == 'M':
                id1.append(count)
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        # id1 = [z - 1 for i, z in enumerate(id1)]
        id1 = [z for i, z in enumerate(id1)]
        for x in range(id1[0]):
            self.slice1.add_module(str(x), features[x])
        for x in range(id1[0], id1[1]):
            self.slice2.add_module(str(x), features[x])
        for x in range(id1[1], id1[2]):
            self.slice3.add_module(str(x), features[x])
        for x in range(id1[2], id1[3]):
            self.slice4.add_module(str(x), features[x])
        for x in range(id1[3], id1[4]):
            self.slice5.add_module(str(x), features[x])

        if not pretrained:
            self._initialize_weights()

    def forward(self, x):
        h = self.slice1(x)
        h_relu1 = h
        h = self.slice2(h)
        h_relu2 = h
        h = self.slice3(h)
        h_relu3 = h
        h = self.slice4(h)
        h_relu4 = h
        h = self.slice5(h)
        h_relu5 = h
        vgg_outputs = namedtuple("SideOutputs", ['relu1', 'relu2', 'relu3', 'relu4', 'relu5'])
        # vgg_outputs = namedtuple("SideOutputs", ['layer1', 'layer2', 'layer3', 'layer4'])
        out = vgg_outputs(h_relu1, h_relu2, h_relu3, h_relu4, h_relu5)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg11(backbone='vgg11', pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    kwargs['cfg_key'] = 'A'
    kwargs['backbone'] = backbone
    kwargs['pretrained'] = pretrained
    model = VGG(make_layers(cfg['A']), **kwargs)
    return model


def vgg11_bn(backbone='vgg11_bn', pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    kwargs['cfg_key'] = 'A'
    kwargs['batch_norm'] = True
    kwargs['backbone'] = "vgg11_bn"
    kwargs['pretrained'] = pretrained
    model = VGG(make_layers(cfg['A'], batch_norm=True), **kwargs)
    return model


def vgg13(backbone='vgg13', pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    kwargs['cfg_key'] = 'B'
    kwargs['backbone'] = backbone
    kwargs['pretrained'] = pretrained
    model = VGG(make_layers(cfg['B']), **kwargs)
    return model


def vgg13_bn(backbone='vgg13_bn', pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    kwargs['cfg_key'] = 'B'
    kwargs['batch_norm'] = True
    kwargs['backbone'] = "vgg13_bn"
    kwargs['pretrained'] = pretrained
    model = VGG(make_layers(cfg['B'], batch_norm=True), **kwargs)
    return model


def vgg16(backbone='vgg16', pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    kwargs['cfg_key'] = 'D'
    kwargs['backbone'] = backbone
    kwargs['pretrained'] = pretrained
    model = VGG(make_layers(cfg['D']), **kwargs)
    return model


def vgg16_bn(backbone='vgg16_bn', pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    kwargs['cfg_key'] = 'D'
    kwargs['batch_norm'] = True
    kwargs['backbone'] = "vgg16_bn"
    kwargs['pretrained'] = pretrained
    model = VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)
    return model


def vgg19(backbone='vgg19', pretrained=False, **kwargs):
    """VGG 19-layer model (configuration "E")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    kwargs['cfg_key'] = 'E'
    kwargs['backbone'] = backbone
    kwargs['pretrained'] = pretrained
    model = VGG(make_layers(cfg['E']), **kwargs)
    return model


def vgg19_bn(backbone='vgg19_bn', pretrained=False, **kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    kwargs['cfg_key'] = 'E'
    kwargs['batch_norm'] = True
    kwargs['backbone'] = "vgg19_bn"
    kwargs['pretrained'] = pretrained
    model = VGG(make_layers(cfg['E'], batch_norm=True), **kwargs)
    return model

