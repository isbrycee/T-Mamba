# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   yykzhjh@163.com
@DateTime :   2023/5/19 16:20
@Version  :   1.0
@License  :   (C)Copyright 2023
"""
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../"))

import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

from lib.models.modules.ResBlock import ResBlock



class DenseASPPUNet(nn.Module):
    def __init__(self, in_channels: int = 1, classes: int = 1):
        super(DenseASPPUNet, self).__init__()
        self.max_pool = nn.MaxPool3d(kernel_size=2)
        self.up_sample = nn.Upsample(scale_factor=2)

        self.conv1 = ResBlock(1, 64)
        self.conv2 = ResBlock(64, 128)
        self.conv3 = ResBlock(128, 256)

        self.aspp_bridge = _DenseASPPHead(256, norm_layer=nn.BatchNorm3d)

        self.up_conv2 = ResBlock(256 + 128, 128)
        self.up_conv1 = ResBlock(128 + 64, 64)

        self.output_conv = nn.Conv3d(64, classes, kernel_size=3, stride=1, padding=1)

    def forward(self, x):

        x1 = self.conv1(x)
        x2 = self.max_pool(x1)
        x2 = self.conv2(x2)
        x3 = self.max_pool(x2)
        x3 = self.conv3(x3)

        y3 = self.aspp_bridge(x3)

        y2 = self.up_sample(y3)
        y2 = torch.cat((x2, y2), dim=1)
        y2 = self.up_conv2(y2)

        y1 = self.up_sample(y2)
        y1 = torch.cat((x1, y1), dim=1)
        y1 = self.up_conv1(y1)

        out = self.output_conv(y1)

        return out


class _DenseASPPHead(nn.Module):
    def __init__(self, in_channels, norm_layer=nn.BatchNorm3d, norm_kwargs=None, **kwargs):
        super(_DenseASPPHead, self).__init__()
        self.dense_aspp_block = _DenseASPPBlock(in_channels, 64, 32, norm_layer, norm_kwargs)
        self.block = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv3d(in_channels + 3 * 32, in_channels, 1)
        )

    def forward(self, x):
        x = self.dense_aspp_block(x)
        return self.block(x)


class _DenseASPPBlock(nn.Module):
    def __init__(self, in_channels, inter_channels1, inter_channels2, norm_layer=nn.BatchNorm3d, norm_kwargs=None):
        super(_DenseASPPBlock, self).__init__()
        self.aspp_1 = _DenseASPPConv(in_channels, inter_channels1, inter_channels2, 1, 0.1, norm_layer, norm_kwargs)
        self.aspp_2 = _DenseASPPConv(in_channels + inter_channels2 * 1, inter_channels1, inter_channels2, 2, 0.1, norm_layer, norm_kwargs)
        self.aspp_5 = _DenseASPPConv(in_channels + inter_channels2 * 2, inter_channels1, inter_channels2, 5, 0.1, norm_layer, norm_kwargs)

    def forward(self, x):
        aspp1 = self.aspp_1(x)
        x = torch.cat([aspp1, x], dim=1)

        aspp2 = self.aspp_2(x)
        x = torch.cat([aspp2, x], dim=1)

        aspp5 = self.aspp_5(x)
        x = torch.cat([aspp5, x], dim=1)

        return x


class _DenseASPPConv(nn.Sequential):
    def __init__(self, in_channels, inter_channels, out_channels, atrous_rate, drop_rate=0.1, norm_layer=nn.BatchNorm2d, norm_kwargs=None):
        super(_DenseASPPConv, self).__init__()
        self.add_module('conv1', nn.Conv3d(in_channels, inter_channels, 1)),
        self.add_module('bn1', norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs))),
        self.add_module('relu1', nn.ReLU(True)),
        self.add_module('conv2', nn.Conv3d(inter_channels, out_channels, 3, dilation=atrous_rate, padding=atrous_rate)),
        self.add_module('bn2', norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs))),
        self.add_module('relu2', nn.ReLU(True)),
        self.drop_rate = drop_rate

    def forward(self, x):
        features = super(_DenseASPPConv, self).forward(x)
        if self.drop_rate > 0:
            features = F.dropout(features, p=self.drop_rate, training=self.training)
        return features




if __name__ == '__main__':
    x = torch.randn((1, 1, 160, 160, 96))

    model = DenseASPPUNet(1, classes=2)

    y = model(x)

    print(x.size())
    print(y.size())

