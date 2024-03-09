# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   yykzhjh@163.com
@DateTime :   2022/12/2 21:05
@Version  :   1.0
@License  :   (C)Copyright 2022
"""
import torch
import torch.nn as nn


class ConvBlock(torch.nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size,
            stride=1,
            batch_norm=True,
            preactivation=False,
            dim="3d"
    ):
        super().__init__()

        if dim == "3d":
            constant_pad = torch.nn.ConstantPad3d
            conv = torch.nn.Conv3d
            bn = torch.nn.BatchNorm3d
        elif dim == "2d":
            constant_pad = torch.nn.ConstantPad2d
            conv = torch.nn.Conv2d
            bn = torch.nn.BatchNorm2d
        else:
            raise RuntimeError(f"{dim} dimension is error")

        padding = kernel_size - stride
        if padding % 2 != 0:
            pad = constant_pad(
                tuple([padding % 2, padding - padding % 2] * (3 if dim == "3d" else 2)), 0
            )
        else:
            pad = constant_pad(padding // 2, 0)

        if preactivation:
            layers = [
                torch.nn.ReLU(),
                pad,
                conv(
                    in_channels=in_channel,
                    out_channels=out_channel,
                    kernel_size=kernel_size,
                    stride=stride,
                ),
            ]
            if batch_norm:
                layers = [bn(in_channel)] + layers
        else:
            layers = [
                pad,
                conv(
                    in_channels=in_channel,
                    out_channels=out_channel,
                    kernel_size=kernel_size,
                    stride=stride,
                ),
            ]
            if batch_norm:
                layers.append(bn(out_channel))
            layers.append(torch.nn.ReLU())

        self.conv = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


class SingleConvBlock(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size, stride, dim="3d"):
        super(SingleConvBlock, self).__init__()

        if dim == "3d":
            conv = nn.Conv3d
            bn = nn.BatchNorm3d
        elif dim == "2d":
            conv = nn.Conv2d
            bn = nn.BatchNorm2d
        else:
            raise RuntimeError(f"{dim} dimension is error")

        self.conv = nn.Sequential(
            conv(in_channel, out_channel, kernel_size, stride, kernel_size // 2, bias=False),
            bn(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class DepthWiseSeparateConvBlock(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size,
            stride=1,
            batch_norm=True,
            preactivation=False,
            dim="3d"
    ):
        super(DepthWiseSeparateConvBlock, self).__init__()

        if dim == "3d":
            constant_pad = torch.nn.ConstantPad3d
            conv = torch.nn.Conv3d
            bn = torch.nn.BatchNorm3d
        elif dim == "2d":
            constant_pad = torch.nn.ConstantPad2d
            conv = torch.nn.Conv2d
            bn = torch.nn.BatchNorm2d
        else:
            raise RuntimeError(f"{dim} dimension is error")

        padding = kernel_size - stride
        if padding % 2 != 0:
            pad = constant_pad(
                tuple([padding % 2, padding - padding % 2] * (3 if dim == "3d" else 2)), 0
            )
        else:
            pad = constant_pad(padding // 2, 0)

        if preactivation:
            layers = [
                torch.nn.ReLU(),
                pad,
                conv(
                    in_channels=in_channel,
                    out_channels=in_channel,
                    kernel_size=kernel_size,
                    stride=stride,
                    groups=in_channel,
                    bias=False
                ),
                conv(
                    in_channels=in_channel,
                    out_channels=out_channel,
                    kernel_size=1,
                    stride=1,
                    bias=True
                )
            ]
            if batch_norm:
                layers = [bn(in_channel)] + layers
        else:
            layers = [
                pad,
                conv(
                    in_channels=in_channel,
                    out_channels=in_channel,
                    kernel_size=kernel_size,
                    stride=stride,
                    groups=in_channel,
                    bias=False
                ),
                conv(
                    in_channels=in_channel,
                    out_channels=out_channel,
                    kernel_size=1,
                    stride=1,
                    bias=False
                )
            ]
            if batch_norm:
                layers.append(bn(out_channel))
            layers.append(torch.nn.ReLU())

        self.conv = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)
