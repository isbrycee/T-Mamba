# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   yykzhjh@163.com
@DateTime :   2023/5/19 16:03
@Version  :   1.0
@License  :   (C)Copyright 2023
"""
import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(ResBlock, self).__init__()
        self.identity_conv = nn.Conv3d(in_channels=ch_in, out_channels=ch_out, kernel_size=3, stride=1, padding=1, bias=True)
        self.residual_conv = nn.Sequential(
            nn.Conv3d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv3d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(ch_out)
        )

    def forward(self, x):
        identity = self.identity_conv(x)
        x = identity
        residual = self.residual_conv(x)
        return identity + residual









