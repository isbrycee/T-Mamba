import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../"))

import torch
import torch.nn as nn

from lib.models.modules.ConvBlock import ConvBlock


class DenseFeatureStackWithLocalPMFSBlock(nn.Module):

    def __init__(self, in_channel, kernel_size, unit, growth_rate, dim="3d"):
        super(DenseFeatureStackWithLocalPMFSBlock, self).__init__()

        self.conv_units = torch.nn.ModuleList()
        for i in range(unit):
            self.conv_units.append(
                ConvBlock(
                    in_channel=in_channel,
                    out_channel=growth_rate,
                    kernel_size=kernel_size,
                    stride=1,
                    batch_norm=True,
                    preactivation=True,
                    dim=dim
                )
            )
            in_channel += growth_rate

    def forward(self, x):
        stack_feature = None

        for i, conv in enumerate(self.conv_units):
            if stack_feature is None:
                inputs = x
            else:
                inputs = torch.cat([x, stack_feature], dim=1)
            out = conv(inputs)
            if stack_feature is None:
                stack_feature = out
            else:
                stack_feature = torch.cat([stack_feature, out], dim=1)

        return torch.cat([x, stack_feature], dim=1)


class DownSampleWithLocalPMFSBlock(nn.Module):

    def __init__(self, in_channel, base_channel, kernel_size, unit, growth_rate, skip_channel=None, downsample=True, skip=True, dim="3d"):
        super(DownSampleWithLocalPMFSBlock, self).__init__()
        self.skip = skip

        self.downsample = ConvBlock(
            in_channel=in_channel,
            out_channel=base_channel,
            kernel_size=kernel_size,
            stride=(2 if downsample else 1),
            batch_norm=True,
            preactivation=True,
            dim=dim
        )

        self.dfs_with_pmfs = DenseFeatureStackWithLocalPMFSBlock(
            in_channel=base_channel,
            kernel_size=3,
            unit=unit,
            growth_rate=growth_rate,
            dim=dim
        )

        if skip:
            self.skip_conv = ConvBlock(
                in_channel=base_channel + unit * growth_rate,
                out_channel=skip_channel,
                kernel_size=3,
                stride=1,
                batch_norm=True,
                preactivation=True,
                dim=dim
            )

    def forward(self, x):
        x = self.downsample(x)
        x = self.dfs_with_pmfs(x)

        if self.skip:
            x_skip = self.skip_conv(x)
            return x, x_skip
        else:
            return x




if __name__ == '__main__':
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #
    # x = torch.randn((1, 64, 32, 32, 32)).to(device)
    #
    # model = DownSampleWithLocalPMFSBlock(64, 128, 3, 5, 16, skip_channel=24, downsample=True, skip=True, dim="3d").to(device)
    #
    # output = model(x)
    #
    # print(x.size())
    # print(output[0].size())
    # print(output[1].size())

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    x = torch.randn((1, 64, 32, 32)).to(device)

    model = DownSampleWithLocalPMFSBlock(64, 128, 3, 5, 16, skip_channel=24, downsample=True, skip=True, dim="2d").to(device)

    output = model(x)

    print(x.size())
    print(output[0].size())
    print(output[1].size())
