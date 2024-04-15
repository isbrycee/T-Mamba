# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   yykzhjh@163.com
@DateTime :   2022/12/5 23:25
@Version  :   1.0
@License  :   (C)Copyright 2022
"""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../"))

import torch
import torch.nn as nn

from lib.models.modules.ConvBlock import ConvBlock
from lib.models.modules.LocalPMFSBlock import DownSampleWithLocalPMFSBlock
from lib.models.modules.GlobalPMFSBlock import GlobalPMFSBlock_AP_Separate


class PMFSNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=35, dim="3d", scaling_version="TINY",
                 basic_module=DownSampleWithLocalPMFSBlock,
                 global_module=GlobalPMFSBlock_AP_Separate):
        super(PMFSNet, self).__init__()

        self.scaling_version = scaling_version

        if scaling_version == "BASIC":
            base_channels = [24, 48, 64]
            skip_channels = [24, 48, 64]
            units = [5, 10, 10]
            pmfs_ch = 64
        elif scaling_version == "SMALL":
            base_channels = [24, 24, 24]
            skip_channels = [12, 24, 24]
            units = [5, 10, 10]
            pmfs_ch = 48
        elif scaling_version == "TINY":
            base_channels = [24, 24, 24]
            skip_channels = [12, 24, 24]
            units = [3, 5, 5]
            pmfs_ch = 48
        else:
            raise RuntimeError(f"{scaling_version} scaling version is not available")

        if dim == "3d":
            upsample_mode = 'trilinear'
        elif dim == "2d":
            upsample_mode = 'bilinear'
        else:
            raise RuntimeError(f"{dim} dimension is error")
        kernel_sizes = [5, 3, 3]
        growth_rates = [4, 8, 16]
        downsample_channels = [base_channels[i] + units[i] * growth_rates[i] for i in range(len(base_channels))]

        self.down_convs = nn.ModuleList()
        for i in range(3):
            self.down_convs.append(
                basic_module(
                    in_channel=(in_channels if i == 0 else downsample_channels[i - 1]),
                    base_channel=base_channels[i],
                    kernel_size=kernel_sizes[i],
                    skip_channel=skip_channels[i],
                    unit=units[i],
                    growth_rate=growth_rates[i],
                    downsample=True,
                    skip=((i < 2) if scaling_version == "BASIC" else True),
                    dim=dim
                )
            )

        self.Global = global_module(
            in_channels=downsample_channels,
            max_pool_kernels=[4, 2, 1],
            ch=pmfs_ch,
            ch_k=pmfs_ch,
            ch_v=pmfs_ch,
            br=3,
            dim=dim
        )

        if scaling_version == "BASIC":
            self.up2 = torch.nn.Upsample(scale_factor=2, mode=upsample_mode)
            self.up_conv2 = basic_module(in_channel=downsample_channels[2] + skip_channels[1],
                                         base_channel=base_channels[1],
                                         kernel_size=3,
                                         unit=units[1],
                                         growth_rate=growth_rates[1],
                                         downsample=False,
                                         skip=False,
                                         dim=dim)

            self.up1 = torch.nn.Upsample(scale_factor=2, mode=upsample_mode)
            self.up_conv1 = basic_module(in_channel=downsample_channels[1] + skip_channels[0],
                                         base_channel=base_channels[0],
                                         kernel_size=3,
                                         unit=units[0],
                                         growth_rate=growth_rates[0],
                                         downsample=False,
                                         skip=False,
                                         dim=dim)
        else:
            self.bottle_conv = ConvBlock(
                in_channel=downsample_channels[2] + skip_channels[2],
                out_channel=skip_channels[2],
                kernel_size=3,
                stride=1,
                batch_norm=True,
                preactivation=True,
                dim=dim
            )

            self.upsample_1 = torch.nn.Upsample(scale_factor=2, mode=upsample_mode)
            self.upsample_2 = torch.nn.Upsample(scale_factor=4, mode=upsample_mode)

        self.out_conv = ConvBlock(
            in_channel=(downsample_channels[0] if scaling_version == "BASIC" else sum(skip_channels)),
            out_channel=out_channels,
            kernel_size=3,
            stride=1,
            batch_norm=True,
            preactivation=True,
            dim=dim
        )
        self.upsample_out = torch.nn.Upsample(scale_factor=2, mode=upsample_mode)

    def forward(self, x):
        if self.scaling_version == "BASIC":
            x1, x1_skip = self.down_convs[0](x)
            x2, x2_skip = self.down_convs[1](x1)
            x3 = self.down_convs[2](x2)

            d3 = self.Global([x1, x2, x3])

            d2 = self.up2(d3)
            d2 = torch.cat((x2_skip, d2), dim=1)
            d2 = self.up_conv2(d2)
            d1 = self.up1(d2)
            d1 = torch.cat((x1_skip, d1), dim=1)
            d1 = self.up_conv1(d1)

            out = self.out_conv(d1)
            out = self.upsample_out(out)
        else:
            x1, skip1 = self.down_convs[0](x)
            x2, skip2 = self.down_convs[1](x1)
            x3, skip3 = self.down_convs[2](x2)

            x3 = self.Global([x1, x2, x3])
            skip3 = self.bottle_conv(torch.cat([x3, skip3], dim=1))

            skip2 = self.upsample_1(skip2)
            skip3 = self.upsample_2(skip3)

            out = self.out_conv(torch.cat([skip1, skip2, skip3], dim=1))
            out = self.upsample_out(out)

        return out





if __name__ == '__main__':

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dims = ["3d", "2d"]
    channels = [1, 3]

    scaling_versions = ["BASIC", "SMALL", "TINY"]

    xs = [torch.randn((1, 1, 160, 160, 96)).to(device), torch.randn((1, 3, 224, 224)).to(device)]

    for i, dim in enumerate(dims):
        for scaling_version in scaling_versions:
            model = PMFSNet(in_channels=channels[i], out_channels=2, dim=dim, scaling_version=scaling_version).to(device)
            y = model(xs[i])
            print(dim + "-" + scaling_version, ":")
            print(xs[i].size())
            print(y.size())
            print("params: {:.6f}M".format(count_parameters(model)))
