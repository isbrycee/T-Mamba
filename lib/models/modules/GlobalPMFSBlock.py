import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../"))

import torch
import torch.nn as nn

from lib.models.modules.ConvBlock import DepthWiseSeparateConvBlock


class GlobalPMFSBlock_AP_Separate(nn.Module):
    """
    Global polarized multi-scale feature self-attention module using global multi-scale features
    to expand the number of attention points and thus enhance features at each scale,
    replacing standard convolution with depth-wise separable convolution
    """
    def __init__(self, in_channels, max_pool_kernels, ch, ch_k, ch_v, br, dim="3d"):
        """
        Initialize a global polarized multi-scale feature self-attention module that replaces standard convolution with depth-wise separable convolution

        :param in_channels: channels of each scale feature map
        :param max_pool_kernels: sizes of downsample kernels for feature maps at each scale
        :param ch: channel of global uniform feature
        :param ch_k: channel of K
        :param ch_v: channel of V
        :param br: number of branches
        :param dim: dimension
        """
        super(GlobalPMFSBlock_AP_Separate, self).__init__()
        self.ch_bottle = in_channels[-1]
        self.ch = ch
        self.ch_k = ch_k
        self.ch_v = ch_v
        self.br = br
        self.ch_in = self.ch * self.br
        self.dim = dim

        if dim == "3d":
            max_pool = nn.MaxPool3d
            conv = nn.Conv3d
            bn = nn.BatchNorm3d
        elif dim == "2d":
            max_pool = nn.MaxPool2d
            conv = nn.Conv2d
            bn = nn.BatchNorm2d
        else:
            raise RuntimeError(f"{dim} dimension is error")

        self.ch_convs = nn.ModuleList([
            DepthWiseSeparateConvBlock(
                in_channel=in_channel,
                out_channel=self.ch,
                kernel_size=3,
                stride=1,
                batch_norm=True,
                preactivation=True,
                dim=dim
            )
            for in_channel in in_channels
        ])

        self.max_pool_layers = nn.ModuleList([
            max_pool(kernel_size=k, stride=k)
            for k in max_pool_kernels
        ])

        self.ch_Wq = DepthWiseSeparateConvBlock(in_channel=self.ch_in, out_channel=self.ch_in, kernel_size=1, stride=1, batch_norm=True, preactivation=True, dim=dim)
        self.ch_Wk = DepthWiseSeparateConvBlock(in_channel=self.ch_in, out_channel=1, kernel_size=1, stride=1, batch_norm=True, preactivation=True, dim=dim)
        self.ch_Wv = DepthWiseSeparateConvBlock(in_channel=self.ch_in, out_channel=self.ch_in, kernel_size=1, stride=1, batch_norm=True, preactivation=True, dim=dim)
        self.ch_softmax = nn.Softmax(dim=1)
        self.ch_score_conv = conv(self.ch_in, self.ch_in, 1)
        self.ch_layer_norm = (nn.LayerNorm((self.ch_in, 1, 1, 1)) if dim == "3d" else nn.LayerNorm((self.ch_in, 1, 1)))
        self.sigmoid = nn.Sigmoid()

        self.sp_Wq = DepthWiseSeparateConvBlock(in_channel=self.ch_in, out_channel=self.br * self.ch_k, kernel_size=1, stride=1, batch_norm=True, preactivation=True, dim=dim)
        self.sp_Wk = DepthWiseSeparateConvBlock(in_channel=self.ch_in, out_channel=self.br * self.ch_k, kernel_size=1, stride=1, batch_norm=True, preactivation=True, dim=dim)
        self.sp_Wv = DepthWiseSeparateConvBlock(in_channel=self.ch_in, out_channel=self.br * self.ch_v, kernel_size=1, stride=1, batch_norm=True, preactivation=True, dim=dim)
        self.sp_softmax = nn.Softmax(dim=-1)
        self.sp_output_conv = DepthWiseSeparateConvBlock(in_channel=self.br * self.ch_v, out_channel=self.ch_in, kernel_size=1, stride=1, batch_norm=True, preactivation=True, dim=dim)

        self.output_conv = DepthWiseSeparateConvBlock(in_channel=self.ch_in, out_channel=self.ch_bottle, kernel_size=3, stride=1, batch_norm=True, preactivation=True, dim=dim)

    def forward(self, feature_maps):
        max_pool_maps = [
            max_pool_layer(feature_maps[i])
            for i, max_pool_layer in enumerate(self.max_pool_layers)
        ]
        ch_outs = [
            ch_conv(max_pool_maps[i])
            for i, ch_conv in enumerate(self.ch_convs)
        ]
        x = torch.cat(ch_outs, dim=1)

        if self.dim == "3d":
            bs, c, d, h, w = x.size()

            ch_Q = self.ch_Wq(x)  # bs, self.ch_in, d, h, w
            ch_K = self.ch_Wk(x)  # bs, 1, d, h, w
            ch_V = self.ch_Wv(x)  # bs, self.ch_in, d, h, w
            ch_Q = ch_Q.reshape(bs, -1, d * h * w)  # bs, self.ch_in, d*h*w
            ch_K = ch_K.reshape(bs, -1, 1)  # bs, d*h*w, 1
            ch_K = self.ch_softmax(ch_K)  # bs, d*h*w, 1
            Z = torch.matmul(ch_Q, ch_K).unsqueeze(-1).unsqueeze(-1)  # bs, self.ch_in, 1, 1, 1
            ch_score = self.sigmoid(self.ch_layer_norm(self.ch_score_conv(Z)))  # bs, self.ch_in, 1, 1, 1
            ch_out = ch_V * ch_score  # bs, self.ch_in, d, h, w

            sp_Q = self.sp_Wq(ch_out)  # bs, self.br*self.ch_k, d, h, w
            sp_K = self.sp_Wk(ch_out)  # bs, self.br*self.ch_k, d, h, w
            sp_V = self.sp_Wv(ch_out)  # bs, self.br*self.ch_v, d, h, w
            sp_Q = sp_Q.reshape(bs, self.br, self.ch_k, d, h, w).permute(0, 2, 3, 4, 5, 1).reshape(bs, self.ch_k, -1)  # bs, self.ch_k, d*h*w*self.br
            sp_K = sp_K.reshape(bs, self.br, self.ch_k, d, h, w).permute(0, 2, 3, 4, 5, 1).mean(-1).mean(-1).mean(-1).mean(-1).reshape(bs, 1, self.ch_k)  # bs, 1, self.ch_k
            sp_V = sp_V.reshape(bs, self.br, self.ch_k, d, h, w).permute(0, 2, 3, 4, 5, 1)  # bs, self.ch_v, d, h, w, self.br
            sp_K = self.sp_softmax(sp_K)  # bs, 1, self.ch_k
            Z = torch.matmul(sp_K, sp_Q).reshape(bs, 1, d, h, w, self.br)  # bs, 1, d, h, w, self.br
            sp_score = self.sigmoid(Z)  # bs, 1, d, h, w, self.br
            sp_out = sp_V * sp_score  # bs, self.ch_v, d, h, w, self.br
            sp_out = sp_out.permute(0, 5, 1, 2, 3, 4).reshape(bs, self.br * self.ch_v, d, h, w)  # bs, self.br*self.ch_v, d, h, w
            sp_out = self.sp_output_conv(sp_out)  # bs, self.ch_in, d, h, w

            out = self.output_conv(sp_out)
        else:
            bs, c, h, w = x.size()

            ch_Q = self.ch_Wq(x)  # bs, self.ch_in, h, w
            ch_K = self.ch_Wk(x)  # bs, 1, h, w
            ch_V = self.ch_Wv(x)  # bs, self.ch_in, h, w
            ch_Q = ch_Q.reshape(bs, -1, h * w)  # bs, self.ch_in, h*w
            ch_K = ch_K.reshape(bs, -1, 1)  # bs, h*w, 1
            ch_K = self.ch_softmax(ch_K)  # bs, h*w, 1
            Z = torch.matmul(ch_Q, ch_K).unsqueeze(-1)  # bs, self.ch_in, 1, 1
            ch_score = self.sigmoid(self.ch_layer_norm(self.ch_score_conv(Z)))  # bs, self.ch_in, 1, 1
            ch_out = ch_V * ch_score  # bs, self.ch_in, h, w

            sp_Q = self.sp_Wq(ch_out)  # bs, self.br*self.ch_k, h, w
            sp_K = self.sp_Wk(ch_out)  # bs, self.br*self.ch_k, h, w
            sp_V = self.sp_Wv(ch_out)  # bs, self.br*self.ch_v, h, w
            sp_Q = sp_Q.reshape(bs, self.br, self.ch_k, h, w).permute(0, 2, 3, 4, 1).reshape(bs, self.ch_k, -1)  # bs, self.ch_k, h*w*self.br
            sp_K = sp_K.reshape(bs, self.br, self.ch_k, h, w).permute(0, 2, 3, 4, 1).mean(-1).mean(-1).mean(-1).reshape(bs, 1, self.ch_k)  # bs, 1, self.ch_k
            sp_V = sp_V.reshape(bs, self.br, self.ch_k, h, w).permute(0, 2, 3, 4, 1)  # bs, self.ch_v, h, w, self.br
            sp_K = self.sp_softmax(sp_K)  # bs, 1, self.ch_k
            Z = torch.matmul(sp_K, sp_Q).reshape(bs, 1, h, w, self.br)  # bs, 1, h, w, self.br
            sp_score = self.sigmoid(Z)  # bs, 1, h, w, self.br
            sp_out = sp_V * sp_score  # bs, self.ch_v, h, w, self.br
            sp_out = sp_out.permute(0, 4, 1, 2, 3).reshape(bs, self.br * self.ch_v, h, w)  # bs, self.br*self.ch_v, h, w
            sp_out = self.sp_output_conv(sp_out)  # bs, self.ch_in, h, w

            out = self.output_conv(sp_out)
        return out





if __name__ == '__main__':
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #
    # x = [
    #     torch.randn((1, 32, 80, 80, 48)).to(device),
    #     torch.randn((1, 64, 40, 40, 24)).to(device),
    #     torch.randn((1, 128, 20, 20, 12)).to(device),
    # ]
    #
    # model = GlobalPMFSBlock_AP_Separate([32, 64, 128], [4, 2, 1], 64, 64, 64, 3, dim="3d").to(device)
    #
    # output = model(x)
    #
    # print(output.size())

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    x = [
        torch.randn((1, 32, 80, 80)).to(device),
        torch.randn((1, 64, 40, 40)).to(device),
        torch.randn((1, 128, 20, 20)).to(device),
    ]

    model = GlobalPMFSBlock_AP_Separate([32, 64, 128], [4, 2, 1], 64, 64, 64, 3, dim="2d").to(device)

    output = model(x)

    print(output.size())
