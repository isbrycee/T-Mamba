# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   1378453948@qq.com
@DateTime :   2023/11/07 13:42
@Version  :   1.0
@License  :   (C)Copyright 2023
"""
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from lib.models.modules import vgg, resnet_CKDNet


BACKBONE = {
    'vgg16bn': vgg.vgg16_bn,
    'vgg16': vgg.vgg16,
    'vgg19bn': vgg.vgg19_bn,
    'vgg19': vgg.vgg19,
    'resnet101': resnet_CKDNet.ResNet,
    # 'resnet50': resnet.ResNet,
    # 'resnet152': resnet.ResNet,
    # 'resnet34': resnet.ResNet,
}


class DeepLab_Aux(nn.Module):
    def __init__(self, backbone='resnet101', num_classes=1, return_features=False):
        super(DeepLab_Aux, self).__init__()
        self.return_features = return_features
        self.backbone = BACKBONE[backbone](backbone=backbone, pretrained=False)
        self.seg_branch = DeepLabDecoder(backbone=backbone, num_class=num_classes, return_features=return_features)

    def get_backbone_layers(self):
        small_lr_layers = []
        small_lr_layers.append(self.backbone)
        return small_lr_layers

    def forward(self, image, branch='cls'):
        backbone_out = self.backbone(image)
        if self.return_features:
            seg_out, features, aspp = self.seg_branch(backbone_out)
            return seg_out, backbone_out, features, aspp
        seg_out = self.seg_branch(backbone_out)
        return seg_out


class ASPPModule(nn.Module):

    def __init__(self, inplanes, planes, kernel_size, padding, dilation, batchnorm):

        super(ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=1, padding=padding,
                                     dilation=dilation, bias=False)
        self.bn = batchnorm(planes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)


class ASPP(nn.Module):

    def __init__(self, backbone, output_stride, batchnorm):

        super(ASPP, self).__init__()
        if backbone == 'vgg16bn':
            inplanes = 512
        elif backbone == 'resnet101' or backbone == 'resnet152':
            inplanes = 2048
        elif backbone == 'resnet34':
            inplanes = 512
        else:
            raise Exception('Unknown backbone')

        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = ASPPModule(inplanes, 256, 1, padding=0, dilation=dilations[0], batchnorm=batchnorm)
        self.aspp2 = ASPPModule(inplanes, 256, 3, padding=dilations[1], dilation=dilations[1], batchnorm=batchnorm)
        self.aspp3 = ASPPModule(inplanes, 256, 3, padding=dilations[2], dilation=dilations[2], batchnorm=batchnorm)
        self.aspp4 = ASPPModule(inplanes, 256, 3, padding=dilations[3], dilation=dilations[3], batchnorm=batchnorm)

        self.global_average_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                                 nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
                                                 nn.ReLU())
        self.bn_global_average_pool = batchnorm(256)

        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = batchnorm(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(0.5)

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_average_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x5 = self.bn_global_average_pool(x5)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)


class SegASPPDecoder(nn.Module):
    def __init__(self, num_classes, backbone):
        super(SegASPPDecoder, self).__init__()
        if backbone == 'resnet101' or backbone == 'resnet152':
            low_level_inplanes = 256
        elif backbone == 'vgg16bn':
            low_level_inplanes = 256
        elif backbone == 'resnet34':
            low_level_inplanes = 64
        else:
            raise Exception('Unknown backbone')

        self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(48)
        self.relu = nn.ReLU()

        # aspp always gives out 256 planes + 48 from conv1
        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(),
                                       nn.Dropout2d(),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))

    def forward(self, x, low_level_feat):

        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)

        x = F.interpolate(x, low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        second_to_last_features = torch.cat((x, low_level_feat), dim=1)
        x = self.last_conv(second_to_last_features)
        return x, second_to_last_features


class DeepLabDecoder(nn.Module):

    def __init__(self, backbone='vgg16bn', num_class=1, output_stride=16, return_features=False):
        super(DeepLabDecoder, self).__init__()
        batchnorm = nn.BatchNorm2d
        self.backbone = backbone
        self.aspp = ASPP(backbone, output_stride, batchnorm)
        self.decoder = SegASPPDecoder(num_class, backbone)
        self.return_features = return_features
        self.noisy_features = False

    def set_return_features(self, return_features):
        self.return_features = return_features

    def set_noisy_features(self, noisy_features):
        self.noisy_features = noisy_features

    def forward(self, input):
        if self.noisy_features is True:
            noise_input = np.random.normal(loc=0.0, scale=abs(input.mean().cpu().item() * 0.05),
                                           size=input.shape).astype(np.float32)
            input = input + torch.from_numpy(noise_input).cuda()

        if 'vgg' in self.backbone:
            x, low_level_feat = input.relu5, input.relu3
        elif 'resnet' in self.backbone:
            x, low_level_feat = input.layer4, input.layer1
        else:
            raise Exception('Unknown backbone')

        if self.noisy_features is True:
            noise_x = np.random.normal(loc=0.0, scale=abs(x.mean().cpu().item() * 0.5), size=x.shape).astype(np.float32)
            noise_low_level_feat = np.random.normal(loc=0.0, scale=abs(low_level_feat.mean().cpu().item() *
                                                                       0.5), size=low_level_feat.shape).astype(
                np.float32)
            x += torch.from_numpy(noise_x).cuda()
            low_level_feat += torch.from_numpy(noise_low_level_feat).cuda()

        x = self.aspp(x)
        aspp = x
        if self.noisy_features is True:
            noise_x = np.random.normal(loc=0.0, scale=abs(x.mean().cpu().item() * 0.5), size=x.shape).astype(np.float32)
            x += torch.from_numpy(noise_x).cuda()

        low_res_x, features = self.decoder(x, low_level_feat)
        x = F.interpolate(low_res_x, scale_factor=4, mode='bilinear', align_corners=True)
        if self.return_features:
            return x, features, aspp
        return x

