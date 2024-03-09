import torch
import torch.nn as nn


class AttentionUNet3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentionUNet3D, self).__init__()

        self.conv1 = DoubleConvSame3D(c_in=in_channels, c_out=64)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        self.enc1 = Encoder3D(64)
        self.enc2 = Encoder3D(128)
        self.enc3 = Encoder3D(256)
        self.enc4 = Encoder3D(512)

        self.conv5 = DoubleConvSame3D(c_in=512, c_out=1024)

        self.attn1 = AttentionBlock3D(1024, 512)
        self.attn2 = AttentionBlock3D(512, 256)
        self.attn3 = AttentionBlock3D(256, 128)
        self.attn4 = AttentionBlock3D(128, 64)

        self.attndeco1 = AttentionDecoder3D(1024)
        self.attndeco2 = AttentionDecoder3D(512)
        self.attndeco3 = AttentionDecoder3D(256)
        self.attndeco4 = AttentionDecoder3D(128)

        self.conv_1x1 = nn.Conv3d(in_channels=64, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        """ENCODER"""

        c1 = self.conv1(x)
        p1 = self.pool(c1)

        c2, p2 = self.enc1(p1)
        c3, p3 = self.enc2(p2)
        c4, p4 = self.enc3(p3)

        """BOTTLE-NECK"""

        c5 = self.conv5(p4)

        """DECODER - WITH ATTENTION"""

        att1 = self.attn1(c5, c4)
        uc1 = self.attndeco1(c5, c4, att1)

        att2 = self.attn2(uc1, c3)
        uc2 = self.attndeco2(c4, c3, att2)

        att3 = self.attn3(uc2, c2)
        uc3 = self.attndeco3(c3, c2, att3)

        att4 = self.attn4(uc3, c1)
        uc4 = self.attndeco4(c2, c1, att4)

        outputs = self.conv_1x1(uc4)

        return outputs


class AttentionDecoder3D(nn.Module):
    def __init__(self, in_channels):
        super(AttentionDecoder3D, self).__init__()

        self.up_conv = DoubleConvSame3D(c_in=in_channels, c_out=in_channels // 2)
        self.up = nn.ConvTranspose3d(
            in_channels=in_channels,
            out_channels=in_channels // 2,
            kernel_size=2,
            stride=2,
        )

    def forward(self, conv1, conv2, attn):
        up = self.up(conv1)
        mult = torch.multiply(attn, up)
        cat = torch.cat([mult, conv2], dim=1)
        uc = self.up_conv(cat)

        return uc


class AttentionBlock3D(nn.Module):
    """
    Class for creating Attention module
    Takes in gating signal `g` and `x`
    """

    def __init__(self, g_chl, x_chl):
        super(AttentionBlock3D, self).__init__()

        inter_shape = x_chl // 4

        # Conv 1x1 with stride 2 for `x`
        self.conv_x = nn.Conv3d(
            in_channels=x_chl,
            out_channels=inter_shape,
            kernel_size=1,
            stride=2,
        )

        # Conv 1x1 with stride 1 for `g` (gating signal)
        self.conv_g = nn.Conv3d(
            in_channels=g_chl,
            out_channels=inter_shape,
            kernel_size=1,
            stride=1,
        )

        # Conv 1x1 for `psi` the output after `g` + `x`
        self.psi = nn.Conv3d(
            in_channels=2 * inter_shape,
            out_channels=1,
            kernel_size=1,
            stride=1,
        )

        # For upsampling the attention output to size of `x`
        self.upsample = nn.Upsample(scale_factor=2)

    def forward(self, g, x):
        # perform the convs on `x` and `g`
        theta_x = self.conv_x(x)
        gate = self.conv_g(g)

        # `theta_x` + `gate`
        add = torch.cat([gate, theta_x], dim=1)

        # ReLU on the add operation
        relu = torch.relu(add)

        # the 1x1 Conv
        psi = self.psi(relu)

        # Sigmoid the squash the outputs/attention weights
        sig = torch.sigmoid(psi)

        # Upsample to original size of `x` to perform multiplication
        upsample = self.upsample(sig)

        # return the attention weights!
        return upsample


class DoubleConv(nn.Module):
    def __init__(self, c_in, c_out):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=c_out, out_channels=c_out, kernel_size=3),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class DoubleConvSame(nn.Module):
    def __init__(self, c_in, c_out):
        super(DoubleConvSame, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=c_out, out_channels=c_out, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class DoubleConvSame3D(nn.Module):
    def __init__(self, c_in, c_out):
        super(DoubleConvSame3D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(
                in_channels=c_in, out_channels=c_out, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                in_channels=c_out,
                out_channels=c_out,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(self, in_channels):
        super(Encoder, self).__init__()

        self.conv = DoubleConvSame(c_in=in_channels, c_out=in_channels * 2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        c = self.conv(x)
        p = self.pool(c)

        return c, p


class Encoder3D(nn.Module):
    def __init__(self, in_channels):
        super(Encoder3D, self).__init__()

        self.conv = DoubleConvSame3D(c_in=in_channels, c_out=in_channels * 2)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        c = self.conv(x)
        p = self.pool(c)

        return c, p
