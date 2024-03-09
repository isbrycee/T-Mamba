# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   yykzhjh@163.com
@DateTime :   2023/5/15 17:41
@Version  :   1.0
@License  :   (C)Copyright 2023
"""
import os
import torch
import torch.nn as nn



class MultiResUNet3D(nn.Module):
    def __init__(self, in_channels, classes):
        super(MultiResUNet3D, self).__init__()
        self.Maxpool = nn.MaxPool3d(kernel_size=2, stride=2)

        self.mresblock1 = MultiResBlock(in_channels, 32)
        # self.up_conv1 = up_conv(32*2, 32)
        self.res_path1 = ResPath(51, 4)
        self.mresblock2 = MultiResBlock(51, 32 * 2)
        # self.up_conv2 = up_conv(32*4, 32*2)
        self.res_path2 = ResPath(105, 3)
        self.mresblock3 = MultiResBlock(105, 32 * 4)
        # self.up_conv3 = up_conv(32*8, 32*4)
        self.res_path3 = ResPath(212, 2)
        self.mresblock4 = MultiResBlock(212, 32 * 8)
        self.up_5 = nn.ConvTranspose3d(853, 426, 2, 2)
        self.res_path4 = ResPath(426, 1)
        self.mresblock5 = MultiResBlock(426, 32 * 16)
        self.mresblock6 = MultiResBlock(852, 32 * 8)
        self.up_6 = nn.ConvTranspose3d(426, 212, 2, 2)
        self.mresblock7 = MultiResBlock(424, 32 * 4)
        self.up_7 = nn.ConvTranspose3d(212, 105, 2, 2)
        self.obo = conv_block(32, 1, act='sigmoid')
        self.mresblock8 = MultiResBlock(210, 32 * 2)
        self.up_8 = nn.ConvTranspose3d(105, 51, 2, 2)
        self.mresblock9 = MultiResBlock(102, 32)
        self.conv_bn = nn.Sequential(
            nn.Conv3d(51, classes, 3, 1, 1),
            nn.BatchNorm3d(classes)
        )

    def forward(self, x):
        x1 = self.mresblock1(x)  # 1 51 256 256
        pool1 = self.Maxpool(x1)
        self.res_path1(x1)

        x2 = self.mresblock2(pool1)  # 1 105  128 128
        pool2 = self.Maxpool(x2)
        self.res_path2(x2)

        x3 = self.mresblock3(pool2)  # 1 212 64 64
        pool3 = self.Maxpool(x3)
        self.res_path3(x3)

        x4 = self.mresblock4(pool3)  # 1 426 32 32
        pool4 = self.Maxpool(x4)
        self.res_path4(x4)

        x5 = self.mresblock5(pool4)  # 1 853 16 16
        up5 = self.up_5(x5)
        cat5 = torch.cat((up5, x4), dim=1)
        x6 = self.mresblock6(cat5)  # 1 426 32 32

        up6 = self.up_6(x6)
        cat6 = torch.cat((up6, x3), dim=1)
        x7 = self.mresblock7(cat6)

        up7 = self.up_7(x7)
        cat7 = torch.cat((up7, x2), dim=1)
        x8 = self.mresblock8(cat7)

        up8 = self.up_8(x8)
        cat8 = torch.cat((up8, x1), dim=1)
        x9 = self.mresblock9(cat8)
        out = self.conv_bn(x9)
        return out


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size=3, stride=1, padding=1, act='relu'):
        super(conv_block, self).__init__()
        if act == None:
            self.conv = nn.Sequential(
                nn.Conv3d(ch_in, ch_out, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.BatchNorm3d(ch_out)
            )
        elif act == 'relu':
            self.conv = nn.Sequential(
                nn.Conv3d(ch_in, ch_out, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.BatchNorm3d(ch_out),
                nn.ReLU(inplace=True)
            )
        elif act == 'sigmoid':
            self.conv = nn.Sequential(
                nn.Conv3d(ch_in, ch_out, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.BatchNorm3d(ch_out),
                nn.Sigmoid()
            )

    def forward(self, x):
        x = self.conv(x)
        return x


class res_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(res_block, self).__init__()
        self.res = conv_block(ch_in, ch_out, 1, 1, 0, None)
        self.main = conv_block(ch_in, ch_out)
        self.bn = nn.BatchNorm3d(ch_in)

    def forward(self, x):
        res_x = self.res(x)

        main_x = self.main(x)
        out = res_x.add(main_x)
        out = nn.ReLU(inplace=True)(out)
        out = self.bn(out)
        return out


class ResPath(nn.Module):
    def __init__(self, ch, stage):
        super(ResPath, self).__init__()
        self.stage = stage
        self.block = res_block(ch, ch)

    def forward(self, x):
        out = self.block(x)
        for i in range(self.stage - 1):
            out = self.block(out)


class MultiResBlock(nn.Module):
    def __init__(self, in_ch, U, alpha=1.67):
        super(MultiResBlock, self).__init__()
        self.W = alpha * U
        self.one_ch = conv_block(in_ch, 1)
        self.residual_layer = conv_block(1, int(self.W * 0.167) + int(self.W * 0.333) + int(self.W * 0.5), 1, 1, 0,
                                         act=None)
        self.conv3x3 = conv_block(1, int(self.W * 0.167))
        self.conv5x5 = conv_block(int(self.W * 0.167), int(self.W * 0.333))
        self.conv7x7 = conv_block(int(self.W * 0.333), int(self.W * 0.5))
        self.relu = nn.ReLU(inplace=True)
        self.batchnorm_1 = nn.BatchNorm3d(int(self.W * 0.167) + int(self.W * 0.333) + int(self.W * 0.5))
        self.batchnorm_2 = nn.BatchNorm3d(int(self.W * 0.167) + int(self.W * 0.333) + int(self.W * 0.5))

    def forward(self, x):
        x = self.one_ch(x)
        res = self.residual_layer(x)
        sbs = self.conv3x3(x)
        obo = self.conv5x5(sbs)
        cbc = self.conv7x7(obo)
        all_t = torch.cat((sbs, obo, cbc), 1)
        all_t_b = self.batchnorm_1(all_t)
        out = all_t_b.add(res)
        out = self.relu(out)
        out = self.batchnorm_2(out)
        return out


