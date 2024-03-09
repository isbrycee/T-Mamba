import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../"))

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable

# from lib.models.modules.GlobalPMFSBlock import GlobalPMFSBlock_AP_Separate


class BCDUNet(nn.Module):
    def __init__(self, input_dim=3, output_dim=3, num_filter=64, frame_size=(256, 256), bidirectional=False, norm='instance'):
        super(BCDUNet, self).__init__()
        self.num_filter = num_filter
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)
        self.frame_size = np.array(frame_size)

        if norm == 'instance':
            norm_layer = nn.InstanceNorm2d
        else:
            norm_layer = nn.BatchNorm2d

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                norm_layer(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                norm_layer(out_channels),
                nn.ReLU(inplace=True)
            )

        self.conv1 = conv_block(input_dim, num_filter)
        self.conv2 = conv_block(num_filter, num_filter * 2)
        self.conv3 = conv_block(num_filter * 2, num_filter * 4)
        self.conv4 = conv_block(num_filter * 4, num_filter * 8)

        # self.Global = GlobalPMFSBlock_AP_Separate(
        #     in_channels=[64, 128, 256, 512],
        #     max_pool_kernels=[4, 2, 1, 1],
        #     ch=64,
        #     ch_k=64,
        #     ch_v=64,
        #     br=4
        # )

        self.upconv3 = nn.ConvTranspose2d(num_filter * 8, num_filter * 4, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(num_filter * 4, num_filter * 2, kernel_size=2, stride=2)
        self.upconv1 = nn.ConvTranspose2d(num_filter * 2, num_filter, kernel_size=2, stride=2)

        self.conv3m = conv_block(num_filter * 8, num_filter * 4)
        self.conv2m = conv_block(num_filter * 4, num_filter * 2)
        self.conv1m = conv_block(num_filter * 2, num_filter)

        self.conv0 = nn.Conv2d(num_filter, output_dim, kernel_size=1)

        if bidirectional:
            self.clstm1 = ConvBLSTM(num_filter * 4, num_filter * 2, (3, 3), (1, 1), 'tanh', list(self.frame_size // 4))
            self.clstm2 = ConvBLSTM(num_filter * 2, num_filter, (3, 3), (1, 1), 'tanh', list(self.frame_size // 2))
            self.clstm3 = ConvBLSTM(num_filter, num_filter // 2, (3, 3), (1, 1), 'tanh', list(self.frame_size))
        else:
            self.clstm1 = ConvLSTM(num_filter * 4, num_filter * 2, (3, 3), (1, 1), 'tanh', list(self.frame_size // 4))
            self.clstm2 = ConvLSTM(num_filter * 2, num_filter, (3, 3), (1, 1), 'tanh', list(self.frame_size // 2))
            self.clstm3 = ConvLSTM(num_filter, num_filter // 2, (3, 3), (1, 1), 'tanh', list(self.frame_size))

    def forward(self, x):
        N = self.frame_size
        conv1 = self.conv1(x)
        pool1 = self.maxpool(conv1)
        conv2 = self.conv2(pool1)
        pool2 = self.maxpool(conv2)
        conv3 = self.conv3(pool2)
        pool3 = self.maxpool(conv3)
        conv4 = self.conv4(pool3)

        # conv4 = self.Global([pool1, pool2, pool3, conv4])

        upconv3 = self.upconv3(conv4)
        concat3 = torch.cat((conv3, upconv3), 1)
        conv3m = self.conv3m(concat3)

        upconv2 = self.upconv2(conv3m)
        concat2 = torch.cat((conv2, upconv2), 1)
        conv2m = self.conv2m(concat2)

        upconv1 = self.upconv1(conv2m)
        concat1 = torch.cat((conv1, upconv1), 1)
        conv1m = self.conv1m(concat1)

        conv0 = self.conv0(conv1m)

        return conv0


class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, activation, frame_size):
        super(ConvLSTMCell, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if activation == "tanh":
            self.activation = torch.tanh
        elif activation == "relu":
            self.activation = torch.relu

        self.conv = nn.Conv2d(
            in_channels=in_channels + out_channels,
            out_channels=4 * out_channels,
            kernel_size=kernel_size,
            padding=padding)

        self.W_ci = nn.Parameter(torch.Tensor(out_channels, *frame_size))
        self.W_co = nn.Parameter(torch.Tensor(out_channels, *frame_size))
        self.W_cf = nn.Parameter(torch.Tensor(out_channels, *frame_size))

        # Initialize weights using Xavier initialization
        nn.init.xavier_uniform_(self.W_ci)
        nn.init.xavier_uniform_(self.W_co)
        nn.init.xavier_uniform_(self.W_cf)

    def forward(self, X, H_prev, C_prev):
        conv_output = self.conv(torch.cat([X, H_prev], dim=1))
        i_conv, f_conv, C_conv, o_conv = torch.chunk(conv_output, chunks=4, dim=1)

        input_gate = torch.sigmoid(i_conv + self.W_ci * C_prev)
        forget_gate = torch.sigmoid(f_conv + self.W_cf * C_prev)

        # Current Cell output
        C = forget_gate * C_prev + input_gate * self.activation(C_conv)

        output_gate = torch.sigmoid(o_conv + self.W_co * C)

        # Current Hidden State
        H = output_gate * self.activation(C)

        return H, C


class ConvLSTM(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, activation, frame_size, return_sequence=False):
        super(ConvLSTM, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.out_channels = out_channels
        self.return_sequence = return_sequence

        # We will unroll this over time steps
        self.convLSTMcell = ConvLSTMCell(in_channels, out_channels, kernel_size, padding, activation, frame_size)

    def forward(self, X):
        # X is a frame sequence (batch_size, seq_len, num_channels, height, width)

        # Get the dimensions
        batch_size, seq_len, channels, height, width = X.size()

        # Initialize output
        output = torch.zeros(batch_size, seq_len, self.out_channels, height, width, device=self.device)

        # Initialize Hidden State
        H = torch.zeros(batch_size, self.out_channels, height, width, device=self.device)

        # Initialize Cell Input
        C = torch.zeros(batch_size, self.out_channels, height, width, device=self.device)

        # Unroll over time steps
        for time_step in range(seq_len):
            H, C = self.convLSTMcell(X[:, time_step, ...], H, C)
            output[:, time_step, ...] = H

        if not self.return_sequence:
            output = torch.squeeze(output[:, -1, ...], dim=1)

        return output


class ConvBLSTM(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size, padding, activation, frame_size, return_sequence=False):
        super(ConvBLSTM, self).__init__()
        self.return_sequence = return_sequence
        self.forward_cell = ConvLSTM(in_channels, out_channels // 2,
                                     kernel_size, padding, activation, frame_size, return_sequence=True)
        self.backward_cell = ConvLSTM(in_channels, out_channels // 2,
                                      kernel_size, padding, activation, frame_size, return_sequence=True)

    def forward(self, x):
        y_out_forward = self.forward_cell(x)
        reversed_idx = list(reversed(range(x.shape[1])))
        y_out_reverse = self.backward_cell(x[:, reversed_idx, ...])[:, reversed_idx, ...]
        output = torch.cat((y_out_forward, y_out_reverse), dim=2)
        if not self.return_sequence:
            output = torch.squeeze(output[:, -1, ...], dim=1)
        return output


if __name__ == '__main__':
    x = torch.rand((1, 3, 224, 224)).to("cuda:0")

    model = BCDUNet(input_dim=3, output_dim=2, frame_size=(224, 224)).to("cuda:0")

    y = model(x)

    print(x.size())
    print(y.size())
