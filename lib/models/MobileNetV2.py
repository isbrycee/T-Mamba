# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   1378453948@qq.com
@DateTime :   2023/10/26 01:39
@Version  :   1.0
@License  :   (C)Copyright 2023
"""
import torch
import torch.nn as nn
import math
# from lib.models.modules.GlobalPMFSBlock import GlobalPMFSBlock_AP_Separate
from lib.models.modules.ConvBlock import ConvBlock


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def make_divisible(x, divisible_by=8):
    import numpy as np
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, in_channels=3, out_channels=1000, input_size=224, width_mult=1.):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        self.skip_channels = [16, 24, 32, 96, 1280]
        self.skip_indices = [2, 4, 7, 14, 19]

        # building first layer
        assert input_size % 32 == 0
        # input_channel = make_divisible(input_channel * width_mult)  # first channel is always 32!
        self.last_channel = make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(in_channels, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = make_divisible(c * width_mult) if t > 1 else c
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        self.global_pmfs = GlobalPMFSBlock_AP(self.skip_channels, [16, 8, 4, 2, 1], 64, 64, 64, 5)

        self.upsample_1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsample_2 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upsample_3 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.upsample_4 = nn.Upsample(scale_factor=16, mode='bilinear')

        self.out_conv = ConvBlock(
            in_channel=sum(self.skip_channels),
            out_channel=out_channels,
            kernel_size=3,
            stride=1,
            batch_norm=True,
            preactivation=True,
        )

        self.upsample_out = nn.Upsample(scale_factor=2, mode='bilinear')


    def forward(self, x):
        x1 = self.features[:self.skip_indices[0]](x)
        x2 = self.features[self.skip_indices[0]:self.skip_indices[1]](x1)
        x3 = self.features[self.skip_indices[1]:self.skip_indices[2]](x2)
        x4 = self.features[self.skip_indices[2]:self.skip_indices[3]](x3)
        x5 = self.features[self.skip_indices[3]:self.skip_indices[4]](x4)

        x5 = self.global_pmfs([x1, x2, x3, x4, x5])

        x2 = self.upsample_1(x2)
        x3 = self.upsample_2(x3)
        x4 = self.upsample_3(x4)
        x5 = self.upsample_4(x5)

        out = self.out_conv(torch.cat([x1, x2, x3, x4, x5], dim=1))
        out = self.upsample_out(out)

        return out





if __name__ == '__main__':

    net = MobileNetV2(in_channels=3, out_channels=2, input_size=224, width_mult=1.).to("cuda:0")

    x = torch.rand((4, 3, 224, 224)).to("cuda:0")

    y = net(x)

    print(x.size())
    print(y.size())











