import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../"))

import torch
import torch.nn as nn

from lib.models.modules.modules import conv_block, UpCat, UpCatconv, UnetDsv3, UnetGridGatingSignal3
from lib.models.modules.grid_attention_layer import GridAttentionBlock2D, MultiAttentionBlock
from lib.models.modules.channel_attention_layer import SE_Conv_Block
from lib.models.modules.scale_attention_layer import scale_atten_convblock
from lib.models.modules.nonlocal_layer import NONLocalBlock2D
import numpy as np
from scipy import ndimage

# from lib.models.modules.GlobalPMFSBlock import GlobalPMFSBlock_AP_Separate


class Comprehensive_Atten_Unet(nn.Module):
    def __init__(self, in_ch=3, n_classes=2, out_size=(224, 224), feature_scale=4, is_deconv=True, is_batchnorm=True,
                 nonlocal_mode='concatenation', attention_dsample=(1, 1)):
        super(Comprehensive_Atten_Unet, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_ch
        self.num_classes = n_classes
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.out_size = out_size

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = conv_block(self.in_channels, filters[0])
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv2 = conv_block(filters[0], filters[1])
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv3 = conv_block(filters[1], filters[2])
        self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv4 = conv_block(filters[2], filters[3], drop_out=True)
        self.maxpool4 = nn.MaxPool2d(kernel_size=(2, 2))

        self.center = conv_block(filters[3], filters[4], drop_out=True)

        # self.Global = GlobalPMFSBlock_AP_Separate(
        #     in_channels=[16, 32, 64, 128, 256],
        #     max_pool_kernels=[8, 4, 2, 1, 1],
        #     ch=64,
        #     ch_k=64,
        #     ch_v=64,
        #     br=5
        # )

        # attention blocks
        # self.attentionblock1 = GridAttentionBlock2D(in_channels=filters[0], gating_channels=filters[1],
        #                                             inter_channels=filters[0])
        self.attentionblock2 = MultiAttentionBlock(in_size=filters[1], gate_size=filters[2], inter_size=filters[1],
                                                   nonlocal_mode=nonlocal_mode, sub_sample_factor=attention_dsample)
        self.attentionblock3 = MultiAttentionBlock(in_size=filters[2], gate_size=filters[3], inter_size=filters[2],
                                                   nonlocal_mode=nonlocal_mode, sub_sample_factor=attention_dsample)
        self.nonlocal4_2 = NONLocalBlock2D(in_channels=filters[4], inter_channels=filters[4] // 4)

        # upsampling
        self.up_concat4 = UpCat(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = UpCat(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = UpCat(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = UpCat(filters[1], filters[0], self.is_deconv)
        self.up4 = SE_Conv_Block(filters[4], filters[3], drop_out=True)
        self.up3 = SE_Conv_Block(filters[3], filters[2])
        self.up2 = SE_Conv_Block(filters[2], filters[1])
        self.up1 = SE_Conv_Block(filters[1], filters[0])

        # deep supervision
        self.dsv4 = UnetDsv3(in_size=filters[3], out_size=4, scale_factor=self.out_size)
        self.dsv3 = UnetDsv3(in_size=filters[2], out_size=4, scale_factor=self.out_size)
        self.dsv2 = UnetDsv3(in_size=filters[1], out_size=4, scale_factor=self.out_size)
        self.dsv1 = nn.Conv2d(in_channels=filters[0], out_channels=4, kernel_size=1)

        self.scale_att = scale_atten_convblock(in_size=16, out_size=4)
        # final conv (without any concat)
        self.final = nn.Sequential(nn.Conv2d(4, n_classes, kernel_size=1), nn.Softmax2d())

    def forward(self, inputs):
        # Feature Extraction
        conv1 = self.conv1(inputs)  # [16, 3, 224, 300]-->[16, 16, 224, 300]
        maxpool1 = self.maxpool1(conv1)  # [16, 16, 112, 150]

        conv2 = self.conv2(maxpool1)  # [16, 32, 112, 150]
        maxpool2 = self.maxpool2(conv2)  # [16, 32, 56, 75]

        conv3 = self.conv3(maxpool2)  # [16, 64, 56, 75]
        maxpool3 = self.maxpool3(conv3)  # [16, 64, 28, 37]

        conv4 = self.conv4(maxpool3)  # [16, 128, 28, 37]
        maxpool4 = self.maxpool4(conv4)  # [16, 128, 14, 18]

        # Gating Signal Generation
        center = self.center(maxpool4)  # [16, 256, 14, 18]

        # center = self.Global([maxpool1, maxpool2, maxpool3, maxpool4, center])

        # Attention Mechanism
        # Upscaling Part (Decoder)
        up4 = self.up_concat4(conv4, center)  # [16, 256, 28, 37]
        g_conv4 = self.nonlocal4_2(up4)  # [16, 256, 28, 37]

        up4, att_weight4 = self.up4(g_conv4)  # [16, 128, 28, 37]
        g_conv3, att3 = self.attentionblock3(conv3, up4)  # [16, 64, 56, 75]

        atten3_map = att3.cpu().detach().numpy().astype(float)
        atten3_map = ndimage.interpolation.zoom(atten3_map, [1.0, 1.0, self.out_size[0] / atten3_map.shape[2],
                                                             self.out_size[1] / atten3_map.shape[3]], order=0)

        up3 = self.up_concat3(g_conv3, up4)
        up3, att_weight3 = self.up3(up3)  # [16, 64, 56, 75]
        g_conv2, att2 = self.attentionblock2(conv2, up3)  # [16, 32, 112, 150]

        atten2_map = att2.cpu().detach().numpy().astype(float)
        atten2_map = ndimage.interpolation.zoom(atten2_map, [1.0, 1.0, self.out_size[0] / atten2_map.shape[2],
                                                             self.out_size[1] / atten2_map.shape[3]], order=0)

        up2 = self.up_concat2(g_conv2, up3)
        up2, att_weight2 = self.up2(up2)  # [16, 32, 112, 150]
        # g_conv1, att1 = self.attentionblock1(conv1, up2)

        # atten1_map = att1.cpu().detach().numpy().astype(float)
        # atten1_map = ndimage.interpolation.zoom(atten1_map, [1.0, 1.0, 224 / atten1_map.shape[2],
        #                                                      300 / atten1_map.shape[3]], order=0)
        up1 = self.up_concat1(conv1, up2)
        up1, att_weight1 = self.up1(up1)  # [16, 16, 224, 300]

        # Deep Supervision
        dsv4 = self.dsv4(up4)  # [16, 4, 224, 300]
        dsv3 = self.dsv3(up3)  # [16, 4, 224, 300]
        dsv2 = self.dsv2(up2)  # [16, 4, 224, 300]
        dsv1 = self.dsv1(up1)  # [16, 4, 224, 300]
        dsv_cat = torch.cat([dsv1, dsv2, dsv3, dsv4], dim=1)  # [16, 16, 224, 300]
        out = self.scale_att(dsv_cat)  # [16, 4, 224, 300]

        out = self.final(out)

        return out


if __name__ == '__main__':
    x = torch.rand((1, 3, 224, 224)).to("cuda:0")

    model = Comprehensive_Atten_Unet(in_ch=3, n_classes=2).to("cuda:0")

    y = model(x)

    print(x.size())
    print(y.size())
