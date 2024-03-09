# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   1378453948@qq.com
@DateTime :   2023/11/07 13:34
@Version  :   1.0
@License  :   (C)Copyright 2023
"""
"""
Ms RED network.
"""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../"))
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.models.modules.modules import UnetDsv3
from lib.models.modules.scale_attention_layer_softpool import scale_atten_convblock_softpool

# from lib.models.modules.GlobalPMFSBlock import GlobalPMFSBlock_AP_Separate


class Ms_red_v1(nn.Module):
    def __init__(self, classes, channels, out_size=(224, 224)):
        """
        :param classes: the object classes number.
        :param channels: the channels of the input image.
        """
        super(Ms_red_v1, self).__init__()
        self.out_size = out_size
        self.enc_input = ResEncoder_hs(channels, 32)
        self.encoder1 = RFB_hs(32, 64)
        self.encoder2 = RFB_hs(64, 128)
        self.encoder3 = RFB_hs(128, 256)
        self.encoder4 = RFB_hs_att(256, 512)
        self.downsample = downsample_soft()

        # self.Global = GlobalPMFSBlock_AP_Separate(
        #     in_channels=[32, 64, 128, 256, 512],
        #     max_pool_kernels=[8, 4, 2, 1, 1],
        #     ch=64,
        #     ch_k=64,
        #     ch_v=64,
        #     br=5
        # )

        self.affinity_attention = AffinityAttention(512)
        # self.affinity_attention = AffinityAttention_2(512)
        # self.attention_fuse = nn.Conv2d(512 * 2, 512, kernel_size=1)
        self.decoder4 = RFB_hs_att(512, 256)
        self.decoder3 = RFB_hs(256, 128)
        self.decoder2 = RFB_hs(128, 64)
        self.decoder1 = RFB_hs(64, 32)
        self.deconv4 = deconv(512, 256)
        self.deconv3 = deconv(256, 128)
        self.deconv2 = deconv(128, 64)
        self.deconv1 = deconv(64, 32)

        self.assf_fusion4 = ASFF_ddw(level=0)

        self.dsv4 = UnetDsv3(in_size=256, out_size=4, scale_factor=self.out_size)
        self.dsv3 = UnetDsv3(in_size=128, out_size=4, scale_factor=self.out_size)
        self.dsv2 = UnetDsv3(in_size=64, out_size=4, scale_factor=self.out_size)
        self.dsv1 = nn.Conv2d(in_channels=32, out_channels=4, kernel_size=1)
        self.scale_att = scale_atten_convblock_softpool(in_size=16, out_size=4)
        self.final = nn.Conv2d(4, classes, kernel_size=1)

    def forward(self, x):
        enc_input = self.enc_input(x)  # [16, 3, 224, 320]-->[16, 32, 224, 320]
        down1 = self.downsample(enc_input)  # [16, 32, 112, 160]

        enc1 = self.encoder1(down1)  # [16, 64, 112, 160]
        down2 = self.downsample(enc1)  # [16, 64, 56, 80]

        enc2 = self.encoder2(down2)  # [16, 128, 56, 80]
        down3 = self.downsample(enc2)  # [16, 128, 28, 40]

        enc3 = self.encoder3(down3)  # [16, 256, 28, 40]
        fused1 = self.assf_fusion4(enc3, enc2, enc1, enc_input)
        down4 = self.downsample(fused1)  # [16, 256, 14, 20]

        input_feature = self.encoder4(down4)  # [16, 512, 14, 18]

        # Do Attenttion operations here
        attention = self.affinity_attention(input_feature)  # [16, 512, 14, 18]

        # attention_fuse = self.attention_fuse(torch.cat((input_feature, attention), dim=1))
        attention_fuse = input_feature + attention  # [16, 512, 14, 18]

        # attention_fuse = self.Global([down1, down2, down3, down4, attention_fuse])

        # Do decoder operations here
        up4 = self.deconv4(attention_fuse)  # [16, 256, 28, 36]
        up4 = torch.cat((enc3, up4), dim=1)
        dec4 = self.decoder4(up4)

        up3 = self.deconv3(dec4)
        up3 = torch.cat((enc2, up3), dim=1)
        dec3 = self.decoder3(up3)

        up2 = self.deconv2(dec3)
        up2 = torch.cat((enc1, up2), dim=1)
        dec2 = self.decoder2(up2)

        up1 = self.deconv1(dec2)
        up1 = torch.cat((enc_input, up1), dim=1)
        dec1 = self.decoder1(up1)

        dsv4 = self.dsv4(dec4)  # [16, 4, 224, 320]
        dsv3 = self.dsv3(dec3)
        dsv2 = self.dsv2(dec2)
        dsv1 = self.dsv1(dec1)
        dsv_cat = torch.cat([dsv1, dsv2, dsv3, dsv4], dim=1)  # [16, 16, 224, 320]
        out = self.scale_att(dsv_cat)  # [16, 4, 224, 300]

        out = self.final(out)

        final = torch.sigmoid(out)
        return final


class Ms_red_v2(nn.Module):
    def __init__(self, classes, channels, out_size=(224, 224)):
        """
        :param classes: the object classes number.
        :param channels: the channels of the input image.
        """
        super(Ms_red_v2, self).__init__()
        self.out_size = out_size
        self.enc_input = ResEncoder_hs(channels, 32)
        self.encoder1 = RFB7a_hs(32, 64)
        self.encoder2 = RFB7a_hs(64, 128)
        self.encoder3 = RFB7a_hs(128, 256)
        self.encoder4 = RFB7a_hs_att(256, 512)
        self.downsample = downsample_soft()

        # self.Global = GlobalPMFSBlock_AP_Separate(
        #     in_channels=[32, 64, 128, 256, 512],
        #     max_pool_kernels=[8, 4, 2, 1, 1],
        #     ch=64,
        #     ch_k=64,
        #     ch_v=64,
        #     br=5
        # )

        self.affinity_attention = AffinityAttention(512)
        # self.affinity_attention = AffinityAttention_2(512)
        # self.attention_fuse = nn.Conv2d(512 * 2, 512, kernel_size=1)
        self.decoder4 = RFB7a_hs_att(512, 256)
        self.decoder3 = RFB7a_hs(256, 128)
        self.decoder2 = RFB7a_hs(128, 64)
        self.decoder1 = RFB7a_hs(64, 32)
        self.deconv4 = deconv(512, 256)
        self.deconv3 = deconv(256, 128)
        self.deconv2 = deconv(128, 64)
        self.deconv1 = deconv(64, 32)

        self.assf_fusion4 = ASFF_ddw(level=0)

        self.dsv4 = UnetDsv3(in_size=256, out_size=4, scale_factor=self.out_size)
        self.dsv3 = UnetDsv3(in_size=128, out_size=4, scale_factor=self.out_size)
        self.dsv2 = UnetDsv3(in_size=64, out_size=4, scale_factor=self.out_size)
        self.dsv1 = nn.Conv2d(in_channels=32, out_channels=4, kernel_size=1)
        self.scale_att = scale_atten_convblock_softpool(in_size=16, out_size=4)
        self.final = nn.Conv2d(4, classes, kernel_size=1)

    def forward(self, x):
        enc_input = self.enc_input(x)  # [16, 3, 224, 320]-->[16, 32, 224, 320]
        down1 = self.downsample(enc_input)  # [16, 32, 112, 160]

        enc1 = self.encoder1(down1)  # [16, 64, 112, 160]
        down2 = self.downsample(enc1)  # [16, 64, 56, 80]

        enc2 = self.encoder2(down2)  # [16, 128, 56, 80]
        down3 = self.downsample(enc2)  # [16, 128, 28, 40]

        enc3 = self.encoder3(down3)  # [16, 256, 28, 40]
        fused1 = self.assf_fusion4(enc3, enc2, enc1, enc_input)
        down4 = self.downsample(fused1)  # [16, 256, 14, 20]

        input_feature = self.encoder4(down4)  # [16, 512, 14, 18]

        # Do Attenttion operations here
        attention = self.affinity_attention(input_feature)  # [16, 512, 14, 18]

        # attention_fuse = self.attention_fuse(torch.cat((input_feature, attention), dim=1))
        attention_fuse = input_feature + attention  # [16, 512, 14, 18]

        # attention_fuse = self.Global([down1, down2, down3, down4, attention_fuse])

        # Do decoder operations here
        up4 = self.deconv4(attention_fuse)  # [16, 256, 28, 36]
        up4 = torch.cat((enc3, up4), dim=1)
        dec4 = self.decoder4(up4)

        up3 = self.deconv3(dec4)
        up3 = torch.cat((enc2, up3), dim=1)
        dec3 = self.decoder3(up3)

        up2 = self.deconv2(dec3)
        up2 = torch.cat((enc1, up2), dim=1)
        dec2 = self.decoder2(up2)

        up1 = self.deconv1(dec2)
        up1 = torch.cat((enc_input, up1), dim=1)
        dec1 = self.decoder1(up1)

        dsv4 = self.dsv4(dec4)  # [16, 4, 224, 320]
        dsv3 = self.dsv3(dec3)
        dsv2 = self.dsv2(dec2)
        dsv1 = self.dsv1(dec1)
        dsv_cat = torch.cat([dsv1, dsv2, dsv3, dsv4], dim=1)  # [16, 16, 224, 320]
        out = self.scale_att(dsv_cat)  # [16, 4, 224, 300]

        out = self.final(out)

        final = torch.sigmoid(out)
        return final


def downsample():
    return nn.MaxPool2d(kernel_size=2, stride=2)


def downsample_soft():
    return SoftPooling2D(2, 2)


def deconv(in_channels, out_channels):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)


class SoftPooling2D(torch.nn.Module):
    def __init__(self, kernel_size, strides=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None):
        super(SoftPooling2D, self).__init__()
        self.avgpool = torch.nn.AvgPool2d(kernel_size, strides, padding, ceil_mode, count_include_pad, divisor_override)

    def forward(self, x):
        x_exp = torch.exp(x)
        x_exp_pool = self.avgpool(x_exp)
        x = self.avgpool(x_exp * x)
        return x / x_exp_pool


class HSBlock(nn.Module):
    '''
    替代3x3卷积
    '''

    def __init__(self, in_ch, s=4):
        '''
        特征大小不改变
        :param in_ch: 输入通道
        :param s: 分组数
        '''
        super(HSBlock, self).__init__()
        self.s = s
        self.module_list = nn.ModuleList()
        # 避免无法整除通道数
        in_ch, in_ch_last = (in_ch // s, in_ch // s) if in_ch % s == 0 else (in_ch // s + 1, in_ch % s)
        self.module_list.append(nn.Sequential())
        acc_channels = 0
        for i in range(1, self.s):
            if i == 1:
                channels = in_ch
                acc_channels = channels // 2
            elif i == s - 1:
                channels = in_ch_last + acc_channels
            else:
                channels = in_ch + acc_channels
                acc_channels = channels // 2
            self.module_list.append(self.conv_bn_relu(in_ch=channels, out_ch=channels))

    def conv_bn_relu(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=False)
        )
        return conv_bn_relu

    def forward(self, x):
        x = list(x.chunk(chunks=self.s, dim=1))
        for i in range(1, len(self.module_list)):
            y = self.module_list[i](x[i])
            if i == len(self.module_list) - 1:
                x[0] = torch.cat((x[0], y), 1)
            else:
                y1, y2 = y.chunk(chunks=2, dim=1)
                x[0] = torch.cat((x[0], y1), 1)
                x[i + 1] = torch.cat((x[i + 1], y2), 1)
        return x[0]


class HSBlock_rfb(nn.Module):
    '''
    替代3x3卷积
    '''

    def __init__(self, in_ch, s=4, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        '''
        特征大小不改变
        :param in_ch: 输入通道
        :param s: 分组数
        '''
        super(HSBlock_rfb, self).__init__()
        self.s = s
        self.module_list = nn.ModuleList()
        # 避免无法整除通道数
        in_ch, in_ch_last = (in_ch // s, in_ch // s) if in_ch % s == 0 else (in_ch // s + 1, in_ch % s)
        self.module_list.append(nn.Sequential())
        acc_channels = 0
        for i in range(1, self.s):
            if i == 1:
                channels = in_ch
                acc_channels = channels // 2
            elif i == s - 1:
                channels = in_ch_last + acc_channels
            else:
                channels = in_ch + acc_channels
                acc_channels = channels // 2
            self.module_list.append(self.conv_bn_relu(in_ch=channels, out_ch=channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias))

    def conv_bn_relu(self, in_ch, out_ch, kernel_size, stride, padding, dilation, groups, bias):
        conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=False)
        )
        return conv_bn_relu

    def forward(self, x):
        x = list(x.chunk(chunks=self.s, dim=1))
        for i in range(1, len(self.module_list)):
            y = self.module_list[i](x[i])
            if i == len(self.module_list) - 1:
                x[0] = torch.cat((x[0], y), 1)
            else:
                y1, y2 = y.chunk(chunks=2, dim=1)
                x[0] = torch.cat((x[0], y1), 1)
                x[i + 1] = torch.cat((x[i + 1], y2), 1)
        return x[0]


# conv + bn + relu三剑客，in_planes：输入feature map通道数；out_planes：输出feature map通道数
class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=False) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


# 就是rfbnet论文中的fig 4(a)

class BasicConv_hs(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv_hs, self).__init__()
        self.out_channels = out_planes
        self.conv = HSBlock_rfb(in_planes, s=4, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=False) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class RFB(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale=0.1, visual=1):
        super(RFB, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // 8
        self.branch0 = nn.Sequential(
            BasicConv(in_planes, 2 * inter_planes, kernel_size=1, stride=stride),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=visual, dilation=visual, relu=False)
        )
        self.branch1 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, 2 * inter_planes, kernel_size=(3, 3), stride=stride, padding=(1, 1)),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=visual + 1, dilation=visual + 1, relu=False)
        )
        self.branch2 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, (inter_planes // 2) * 3, kernel_size=3, stride=1, padding=1),  # fig 4(a)中5 x 5 conv拆分成两个3 x 3 conv
            BasicConv((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=3, stride=stride, padding=1),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=2 * visual + 1, dilation=2 * visual + 1, relu=False)
        )

        self.ConvLinear = BasicConv(6 * inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        out = torch.cat((x0, x1, x2), 1)  # concate
        out = self.ConvLinear(out)  # 1 x 1 conv
        short = self.shortcut(x)  # shortcut
        out = out * self.scale + short  # 结合fig 4(a)很容易理解
        out = self.relu(out)  # 最后做一个relu

        return out


class RFB_hs(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale=0.1, visual=1):
        super(RFB_hs, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // 8
        self.branch0 = nn.Sequential(
            BasicConv(in_planes, 2 * inter_planes, kernel_size=1, stride=stride),
            BasicConv_hs(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=visual, dilation=visual, relu=False)
        )
        self.branch1 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, 2 * inter_planes, kernel_size=(3, 3), stride=stride, padding=(1, 1)),
            BasicConv_hs(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=visual + 1, dilation=visual + 1, relu=False)
        )
        self.branch2 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, (inter_planes // 2) * 3, kernel_size=3, stride=1, padding=1),  # fig 4(a)中5 x 5 conv拆分成两个3 x 3 conv
            BasicConv((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=3, stride=stride, padding=1),
            BasicConv_hs(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=2 * visual + 1, dilation=2 * visual + 1, relu=False)
        )

        self.ConvLinear = BasicConv(6 * inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        out = torch.cat((x0, x1, x2), 1)  # concate
        out = self.ConvLinear(out)  # 1 x 1 conv
        short = self.shortcut(x)  # shortcut
        out = out * self.scale + short  # 结合fig 4(a)很容易理解
        out = self.relu(out)  # 最后做一个relu

        return out


class RFB_hs_att(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale=0.1, visual=1):
        super(RFB_hs_att, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // 8
        self.branch0 = nn.Sequential(
            BasicConv(in_planes, 2 * inter_planes, kernel_size=1, stride=stride),
            BasicConv_hs(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=visual, dilation=visual, relu=False),
            AffinityAttention(2 * inter_planes)
        )
        self.branch1 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, 2 * inter_planes, kernel_size=(3, 3), stride=stride, padding=(1, 1)),
            BasicConv_hs(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=visual + 1, dilation=visual + 1, relu=False),
            AffinityAttention(2 * inter_planes)
        )
        self.branch2 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, (inter_planes // 2) * 3, kernel_size=3, stride=1, padding=1),  # fig 4(a)中5 x 5 conv拆分成两个3 x 3 conv
            BasicConv((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=3, stride=stride, padding=1),
            BasicConv_hs(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=2 * visual + 1, dilation=2 * visual + 1, relu=False),
            AffinityAttention(2 * inter_planes)
        )

        self.ConvLinear = BasicConv(6 * inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        out = torch.cat((x0, x1, x2), 1)  # concate
        out = self.ConvLinear(out)  # 1 x 1 conv
        short = self.shortcut(x)  # shortcut
        out = out * self.scale + short  # 结合fig 4(a)很容易理解
        out = self.relu(out)  # 最后做一个relu

        return out


class RFB_att(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale=0.1, visual=1):
        super(RFB_att, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // 8
        self.branch0 = nn.Sequential(
            BasicConv(in_planes, 2 * inter_planes, kernel_size=1, stride=stride),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=visual, dilation=visual, relu=False),
            AffinityAttention(2 * inter_planes)
        )
        self.branch1 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, 2 * inter_planes, kernel_size=(3, 3), stride=stride, padding=(1, 1)),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=visual + 1, dilation=visual + 1, relu=False),
            AffinityAttention(2 * inter_planes)
        )
        self.branch2 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, (inter_planes // 2) * 3, kernel_size=3, stride=1, padding=1),  # fig 4(a)中5 x 5 conv拆分成两个3 x 3 conv
            BasicConv((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=3, stride=stride, padding=1),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=2 * visual + 1, dilation=2 * visual + 1, relu=False),
            AffinityAttention(2 * inter_planes)
        )

        self.ConvLinear = BasicConv(6 * inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        out = torch.cat((x0, x1, x2), 1)  # concate
        out = self.ConvLinear(out)  # 1 x 1 conv
        short = self.shortcut(x)  # shortcut
        out = out * self.scale + short  # 结合fig 4(a)很容易理解
        out = self.relu(out)  # 最后做一个relu

        return out


class RFB7a(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale=0.1, visual=1):
        super(RFB7a, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // 8
        self.branch0 = nn.Sequential(
            BasicConv(in_planes, 2 * inter_planes, kernel_size=1, stride=stride),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=visual, dilation=visual, relu=False)
        )
        self.branch1 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, 2 * inter_planes, kernel_size=(3, 3), stride=stride, padding=(1, 1)),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=visual + 1, dilation=visual + 1, relu=False)
        )
        self.branch2 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, (inter_planes // 2) * 3, kernel_size=3, stride=1, padding=1),  # fig 4(a)中5 x 5 conv拆分成两个3 x 3 conv
            BasicConv((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=3, stride=stride, padding=1),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=2 * visual + 2, dilation=2 * visual + 2, relu=False)
        )
        self.branch3 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, (inter_planes // 2) * 3, kernel_size=3, stride=1, padding=1),  # fig 4(a)中5 x 5 conv拆分成两个3 x 3 conv
            BasicConv((inter_planes // 2) * 3, (inter_planes // 2) * 3, kernel_size=3, stride=1, padding=1),
            BasicConv((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=3, stride=stride, padding=1),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=3 * visual + 3, dilation=3 * visual + 3, relu=False)
        )

        self.ConvLinear = BasicConv(8 * inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        out = torch.cat((x0, x1, x2, x3), 1)  # concate
        out = self.ConvLinear(out)  # 1 x 1 conv
        short = self.shortcut(x)  # shortcut
        out = out * self.scale + short  # 结合fig 4(a)很容易理解
        out = self.relu(out)  # 最后做一个relu

        return out


class RFB7a_hs(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale=0.1, visual=1):
        super(RFB7a_hs, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // 8
        self.branch0 = nn.Sequential(
            BasicConv(in_planes, 2 * inter_planes, kernel_size=1, stride=stride),
            BasicConv_hs(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=visual, dilation=visual, relu=False)
        )
        self.branch1 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, 2 * inter_planes, kernel_size=(3, 3), stride=stride, padding=(1, 1)),
            BasicConv_hs(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=visual + 1, dilation=visual + 1, relu=False)
        )
        self.branch2 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, (inter_planes // 2) * 3, kernel_size=3, stride=1, padding=1),  # fig 4(a)中5 x 5 conv拆分成两个3 x 3 conv
            BasicConv((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=3, stride=stride, padding=1),
            BasicConv_hs(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=2 * visual + 2, dilation=2 * visual + 2, relu=False)
        )
        self.branch3 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, (inter_planes // 2) * 3, kernel_size=3, stride=1, padding=1),  # fig 4(a)中5 x 5 conv拆分成两个3 x 3 conv
            BasicConv((inter_planes // 2) * 3, (inter_planes // 2) * 3, kernel_size=3, stride=1, padding=1),
            BasicConv((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=3, stride=stride, padding=1),
            BasicConv_hs(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=3 * visual + 3, dilation=3 * visual + 3, relu=False)
        )

        self.ConvLinear = BasicConv(8 * inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        out = torch.cat((x0, x1, x2, x3), 1)  # concate
        out = self.ConvLinear(out)  # 1 x 1 conv
        short = self.shortcut(x)  # shortcut
        out = out * self.scale + short  # 结合fig 4(a)很容易理解
        out = self.relu(out)  # 最后做一个relu

        return out


class RFB7a_hs_att(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale=0.1, visual=1):
        super(RFB7a_hs_att, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // 8
        self.branch0 = nn.Sequential(
            BasicConv(in_planes, 2 * inter_planes, kernel_size=1, stride=stride),
            BasicConv_hs(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=visual, dilation=visual, relu=False),
            AffinityAttention(2 * inter_planes)
        )
        self.branch1 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, 2 * inter_planes, kernel_size=(3, 3), stride=stride, padding=(1, 1)),
            BasicConv_hs(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=visual + 1, dilation=visual + 1, relu=False),
            AffinityAttention(2 * inter_planes)
        )
        self.branch2 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, (inter_planes // 2) * 3, kernel_size=3, stride=1, padding=1),  # fig 4(a)中5 x 5 conv拆分成两个3 x 3 conv
            BasicConv((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=3, stride=stride, padding=1),
            BasicConv_hs(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=2 * visual + 2, dilation=2 * visual + 2, relu=False),
            AffinityAttention(2 * inter_planes)
        )
        self.branch3 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, (inter_planes // 2) * 3, kernel_size=3, stride=1, padding=1),  # fig 4(a)中5 x 5 conv拆分成两个3 x 3 conv
            BasicConv((inter_planes // 2) * 3, (inter_planes // 2) * 3, kernel_size=3, stride=1, padding=1),
            BasicConv((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=3, stride=stride, padding=1),
            BasicConv_hs(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=3 * visual + 3, dilation=3 * visual + 3, relu=False),
            AffinityAttention(2 * inter_planes)
        )

        self.ConvLinear = BasicConv(8 * inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        out = torch.cat((x0, x1, x2, x3), 1)  # concate
        out = self.ConvLinear(out)  # 1 x 1 conv
        short = self.shortcut(x)  # shortcut
        out = out * self.scale + short  # 结合fig 4(a)很容易理解
        out = self.relu(out)  # 最后做一个relu

        return out


class ResEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        residual = self.conv1x1(x)
        outc1 = self.conv1(x)
        outb1 = self.bn1(outc1)
        outr1 = self.relu(outb1)
        outc2 = self.conv2(outr1)
        outb2 = self.bn2(outc2)
        out = self.relu(outb2)
        out = out + residual
        out = self.relu(out)
        return out


class ResEncoder_hs(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResEncoder_hs, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = HSBlock(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        residual = self.conv1x1(x)
        outc1 = self.conv1(x)
        outb1 = self.bn1(outc1)
        outr1 = self.relu(outb1)
        outc2 = self.conv2(outr1)
        outb2 = self.bn2(outc2)
        out = self.relu(outb2)
        out = out + residual
        out = self.relu(out)
        return out


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.conv(x)
        return out


class SpatialAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttentionBlock, self).__init__()
        self.query = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(in_channels // 8),
            nn.ReLU(inplace=False)
        )
        self.key = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(in_channels // 8),
            nn.ReLU(inplace=False)
        )
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        :param x: input( BxCxHxW )
        :return: affinity value + x
        """
        B, C, H, W = x.size()
        # compress x: [B,C,H,W]-->[B,H*W,C], make a matrix transpose
        proj_query = self.query(x).view(B, -1, W * H).permute(0, 2, 1)
        proj_key = self.key(x).view(B, -1, W * H)
        affinity = torch.matmul(proj_query, proj_key)
        affinity = self.softmax(affinity)
        proj_value = self.value(x).view(B, -1, H * W)
        weights = torch.matmul(proj_value, affinity.permute(0, 2, 1))
        weights = weights.view(B, C, H, W)
        out = self.gamma * weights + x
        return out


class ChannelAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttentionBlock, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        :param x: input( BxCxHxW )
        :return: affinity value + x
        """
        B, C, H, W = x.size()
        proj_query = x.view(B, C, -1)
        proj_key = x.view(B, C, -1).permute(0, 2, 1)
        affinity = torch.matmul(proj_query, proj_key)
        affinity_new = torch.max(affinity, -1, keepdim=True)[0].expand_as(affinity) - affinity
        affinity_new = self.softmax(affinity_new)
        proj_value = x.view(B, C, -1)
        weights = torch.matmul(affinity_new, proj_value)
        weights = weights.view(B, C, H, W)
        out = self.gamma * weights + x
        return out


class AffinityAttention(nn.Module):
    """ Affinity attention module """

    def __init__(self, in_channels):
        super(AffinityAttention, self).__init__()
        self.sab = SpatialAttentionBlock(in_channels)
        self.cab = ChannelAttentionBlock(in_channels)
        # self.conv1x1 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)

    def forward(self, x):
        """
        sab: spatial attention block
        cab: channel attention block
        :param x: input tensor
        :return: sab + cab
        """
        sab = self.sab(x)
        cab = self.cab(x)
        out = sab + cab
        return out


class AffinityAttention_2(nn.Module):
    """ Affinity attention module """

    def __init__(self, in_channels):
        super(AffinityAttention_2, self).__init__()
        self.sab = SpatialAttentionBlock(in_channels)
        self.cab = ChannelAttentionBlock(in_channels)
        # self.conv1x1 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)

    def forward(self, x):
        """
        sab: spatial attention block
        cab: channel attention block
        :param x: input tensor
        :return: sab + cab
        """
        out = self.sab(x) * x
        out = self.cab(out) * out

        return out


def add_conv(in_ch, out_ch, ksize, stride, leaky=True):
    """
    Add a conv2d / batchnorm / leaky ReLU block.
    Args:
        in_ch (int): number of input channels of the convolution layer.
        out_ch (int): number of output channels of the convolution layer.
        ksize (int): kernel size of the convolution layer.
        stride (int): stride of the convolution layer.
    Returns:
        stage (Sequential) : Sequential layers composing a convolution block.
    """
    stage = nn.Sequential()
    pad = (ksize - 1) // 2
    stage.add_module('conv', nn.Conv2d(in_channels=in_ch,
                                       out_channels=out_ch, kernel_size=ksize, stride=stride,
                                       padding=pad, bias=False))
    stage.add_module('batch_norm', nn.BatchNorm2d(out_ch))
    if leaky:
        stage.add_module('leaky', nn.LeakyReLU(0.1))
    else:
        stage.add_module('relu6', nn.ReLU6(inplace=False))
    return stage


class ASFF_ddw(nn.Module):
    def __init__(self, level, rfb=False, vis=False):
        super(ASFF_ddw, self).__init__()
        self.level = level
        self.dim = [256, 128, 64, 32]
        self.inter_dim = self.dim[self.level]
        if level == 0:
            self.stride_level_1 = add_conv(128, self.inter_dim, 3, 2)
            self.stride_level_2 = nn.Sequential(
                add_conv(64, self.inter_dim, 3, 2),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            self.stride_level_3 = nn.Sequential(
                add_conv(32, 64, 3, 2),
                add_conv(64, self.inter_dim, 3, 2),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            self.expand = add_conv(self.inter_dim, 256, 3, 1)  # ????

        elif level == 1:
            self.stride_level_0 = nn.Sequential(
                add_conv(256, self.inter_dim, 1, 1),
                nn.Upsample(size=(56, 80), mode='bilinear')
            )
            self.stride_level_2 = add_conv(64, self.inter_dim, 3, 2)
            self.stride_level_3 = nn.Sequential(
                add_conv(32, self.inter_dim, 3, 2),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            self.expand = add_conv(self.inter_dim, 64, 3, 1)

        elif level == 2:
            self.stride_level_0 = nn.Sequential(
                add_conv(256, self.inter_dim, 1, 1),
                nn.Upsample(size=(112, 160), mode='bilinear')
            )
            self.stride_level_1 = nn.Sequential(
                add_conv(128, self.inter_dim, 1, 1),
                nn.Upsample(size=(112, 160), mode='bilinear')
            )
            self.stride_level_3 = add_conv(32, self.inter_dim, 3, 2)
            self.expand = add_conv(self.inter_dim, 32, 3, 1)

        elif level == 3:
            self.stride_level_0 = nn.Sequential(
                add_conv(256, self.inter_dim, 1, 1),
                nn.Upsample(size=(224, 320), mode='bilinear')
            )
            self.stride_level_1 = nn.Sequential(
                add_conv(128, self.inter_dim, 1, 1),
                nn.Upsample(size=(224, 320), mode='bilinear')
            )
            self.stride_level_2 = nn.Sequential(
                add_conv(64, self.inter_dim, 1, 1),
                nn.Upsample(size=(224, 320), mode='bilinear')
            )
            self.expand = add_conv(self.inter_dim, 32, 3, 1)

        compress_c = 8 if rfb else 16  # when adding rfb, we use half number of channels to save memory

        self.weight_level_0 = add_conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = add_conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = add_conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_3 = add_conv(self.inter_dim, compress_c, 1, 1)

        self.weight_levels = nn.Conv2d(compress_c * 4, 4, kernel_size=1, stride=1, padding=0)
        self.vis = vis

    def forward(self, x_level_0, x_level_1, x_level_2, x_level_3):
        if self.level == 0:
            level_0_resized = x_level_0
            level_1_resized = self.stride_level_1(x_level_1)
            level_2_resized = self.stride_level_2(x_level_2)
            level_3_resized = self.stride_level_3(x_level_3)

        elif self.level == 1:
            level_0_resized = self.stride_level_0(x_level_0)
            level_1_resized = x_level_1
            level_2_resized = self.stride_level_2(x_level_2)
            level_3_resized = self.stride_level_3(x_level_3)

        elif self.level == 2:
            level_0_resized = self.stride_level_0(x_level_0)
            level_1_resized = self.stride_level_1(x_level_1)
            level_2_resized = x_level_2
            level_3_resized = self.stride_level_3(x_level_3)

        elif self.level == 3:
            level_0_resized = self.stride_level_0(x_level_0)
            level_1_resized = self.stride_level_1(x_level_1)
            level_2_resized = self.stride_level_2(x_level_2)
            level_3_resized = x_level_3

        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        level_2_weight_v = self.weight_level_2(level_2_resized)
        level_3_weight_v = self.weight_level_3(level_3_resized)

        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v, level_3_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = level_0_resized * levels_weight[:, 0:1, :, :] + \
                            level_1_resized * levels_weight[:, 1:2, :, :] + \
                            level_2_resized * levels_weight[:, 2:3, :, :] + \
                            level_3_resized * levels_weight[:, 3:, :, :]

        out = self.expand(fused_out_reduced)

        if self.vis:
            return out, levels_weight, fused_out_reduced.sum(dim=1)
        else:
            return out


class scale_ddw(nn.Module):
    def __init__(self, level, rfb=False, vis=False):
        super(scale_ddw, self).__init__()
        self.level = level
        self.dim = [256, 128, 64, 32]
        self.inter_dim = self.dim[self.level]
        if level == 0:
            self.stride_level_1 = add_conv(128, self.inter_dim, 3, 2)
            self.stride_level_2 = nn.Sequential(
                add_conv(64, self.inter_dim, 3, 2),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            self.stride_level_3 = nn.Sequential(
                add_conv(32, 64, 3, 2),
                add_conv(64, self.inter_dim, 3, 2),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            self.expand = add_conv(self.inter_dim, 256, 3, 1)  # ????

        elif level == 1:
            self.stride_level_0 = nn.Sequential(
                add_conv(256, self.inter_dim, 1, 1),
                nn.Upsample(size=(56, 80), mode='bilinear')
            )
            self.stride_level_2 = add_conv(64, self.inter_dim, 3, 2)
            self.stride_level_3 = nn.Sequential(
                add_conv(32, self.inter_dim, 3, 2),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            self.expand = add_conv(self.inter_dim, 64, 3, 1)

        elif level == 2:
            self.stride_level_0 = nn.Sequential(
                add_conv(256, self.inter_dim, 1, 1),
                nn.Upsample(size=(112, 160), mode='bilinear')
            )
            self.stride_level_1 = nn.Sequential(
                add_conv(128, self.inter_dim, 1, 1),
                nn.Upsample(size=(112, 160), mode='bilinear')
            )
            self.stride_level_3 = add_conv(32, self.inter_dim, 3, 2)
            self.expand = add_conv(self.inter_dim, 32, 3, 1)

        elif level == 3:
            self.stride_level_0 = nn.Sequential(
                add_conv(256, self.inter_dim, 1, 1),
                nn.Upsample(size=(224, 320), mode='bilinear')
            )
            self.stride_level_1 = nn.Sequential(
                add_conv(128, self.inter_dim, 1, 1),
                nn.Upsample(size=(224, 320), mode='bilinear')
            )
            self.stride_level_2 = nn.Sequential(
                add_conv(64, self.inter_dim, 1, 1),
                nn.Upsample(size=(224, 320), mode='bilinear')
            )
            self.expand = add_conv(self.inter_dim, 32, 3, 1)

    def forward(self, x_level_0, x_level_1, x_level_2, x_level_3):
        if self.level == 0:
            level_0_resized = x_level_0
            level_1_resized = self.stride_level_1(x_level_1)
            level_2_resized = self.stride_level_2(x_level_2)
            level_3_resized = self.stride_level_3(x_level_3)

        elif self.level == 1:
            level_0_resized = self.stride_level_0(x_level_0)
            level_1_resized = x_level_1
            level_2_resized = self.stride_level_2(x_level_2)
            level_3_resized = self.stride_level_3(x_level_3)

        elif self.level == 2:
            level_0_resized = self.stride_level_0(x_level_0)
            level_1_resized = self.stride_level_1(x_level_1)
            level_2_resized = x_level_2
            level_3_resized = self.stride_level_3(x_level_3)

        elif self.level == 3:
            level_0_resized = self.stride_level_0(x_level_0)
            level_1_resized = self.stride_level_1(x_level_1)
            level_2_resized = self.stride_level_2(x_level_2)
            level_3_resized = x_level_3

        return level_0_resized, level_1_resized, level_2_resized, level_3_resized


if __name__ == '__main__':
    x = torch.rand((1, 3, 224, 224)).to("cuda:0")

    model = Ms_red_v2(classes=2, channels=3, out_size=(224, 224)).to("cuda:0")

    y = model(x)

    print(x.size())
    print(y.size())
