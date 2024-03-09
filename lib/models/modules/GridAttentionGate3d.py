# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   yykzhjh@163.com
@DateTime :   2022/11/18 17:44
@Version  :   1.0
@License  :   (C)Copyright 2022
"""
import torch
import torch.nn as nn
from torch.nn import functional as F



class GridAttentionGate3d(nn.Module):
    """
    网格注意力门控模块
    reference to "http://arxiv.org/abs/1804.03999"
    """
    def __init__(self, F_l, F_g, F_int=None, mode="concatenation", sub_sample_factor=2):
        """
        定义一个网格注意力门控模块

        :param F_l: 输入特征图的通道数(一般是跳跃连接的特征图通道数)
        :param F_g: 门控特征图的通道数(一般是上采样前的特征图的通道数)
        :param F_int: 中间层特征图的通道数(一般是输入特征图通道数的一半)
        :param mode: 前向传播计算模式
        :param sub_sample_factor: 上层和下层特征图的尺寸比例
        """
        super(GridAttentionGate3d, self).__init__()
        # 定义中间层特征图的通道数
        if F_int is None:
            F_int = F_l // 2
            if F_int == 0:
                F_int = 1
        # 最终结果输出前的一个点卷积
        self.W = nn.Sequential(
            nn.Conv3d(in_channels=F_l, out_channels=F_l, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm3d(F_l)
        )

        # 定义输入信号的门控信号的结合部分,Theta^T * x_ij + Phi^T * gating_signal + bias
        # 定义输入特征图的变换卷积，输入特征图输入上层，尺寸较大，需要下采样
        self.theta = nn.Conv3d(in_channels=F_l, out_channels=F_int, kernel_size=sub_sample_factor,
                               stride=sub_sample_factor, padding=0, bias=False)
        # 定义门控特征图的变换卷积,bias=True等于公式中最后加的bias
        self.phi = nn.Conv3d(in_channels=F_g, out_channels=F_int, kernel_size=1, stride=1, padding=0, bias=True)

        # 定义ψ，将结合后的特征图通道数降为1
        self.psi = nn.Conv3d(in_channels=F_int, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)

        # 根据指定的模式选择不同的函数执行前向传播操作
        if mode == 'concatenation':
            self.operation_function = self._concatenation
        elif mode == 'concatenation_debug':
            self.operation_function = self._concatenation_debug
        elif mode == 'concatenation_residual':
            self.operation_function = self._concatenation_residual
        else:
            raise NotImplementedError('未知的操作函数！')


    def forward(self, x, g):
        output = self.operation_function(x, g)
        return output


    def _concatenation(self, x, g):
        # 获取和检测维度信息
        input_size = x.size()
        bs = input_size[0]
        assert bs == g.size(0)

        # 输入特征图转换
        theta_x = self.theta(x)
        # 获取转换后输入特征图维度
        theta_x_size = theta_x.size()

        # 门控特征图转换并且上采样到和转换后的输入特征图一样的大小
        phi_g = F.interpolate(self.phi(g), size=theta_x_size[2:], mode="trilinear", align_corners=True)
        # 将转换后的输入特征图和，门控特征图相加，经过一个激活函数
        f = F.relu(theta_x + phi_g, inplace=True)

        # 将注意力特征图转换为通道为1，然后经过Sigmoid转换为近似0-1之间的注意力分数矩阵
        sigm_psi_f = torch.sigmoid(self.psi(f))

        # 将注意力分数矩阵上采样到和输入特征图一样大小
        sigm_psi_f = F.interpolate(sigm_psi_f, size=input_size[2:], mode="trilinear", align_corners=True)
        # 将注意力分数矩阵和原输入特征图逐元素相乘
        y = sigm_psi_f.expand_as(x) * x
        # 将增强后的特征图经过转换处理后输出
        W_y = self.W(y)

        return W_y, sigm_psi_f



    def _concatenation_debug(self, x, g):
        # 获取和检测维度信息
        input_size = x.size()
        bs = input_size[0]
        assert bs == g.size(0)

        # 输入特征图转换
        theta_x = self.theta(x)
        # 获取转换后输入特征图维度
        theta_x_size = theta_x.size()

        # 门控特征图转换并且上采样到和转换后的输入特征图一样的大小
        phi_g = F.interpolate(self.phi(g), size=theta_x_size[2:], mode="trilinear", align_corners=True)
        # 将转换后的输入特征图和，门控特征图相加，经过一个激活函数
        f = F.softplus(theta_x + phi_g)

        # 将注意力特征图转换为通道为1，然后经过Sigmoid转换为近似0-1之间的注意力分数矩阵
        sigm_psi_f = torch.sigmoid(self.psi(f))

        # 将注意力分数矩阵上采样到和输入特征图一样大小
        sigm_psi_f = F.interpolate(sigm_psi_f, size=input_size[2:], mode="trilinear", align_corners=True)
        # 将注意力分数矩阵和原输入特征图逐元素相乘
        y = sigm_psi_f.expand_as(x) * x
        # 将增强后的特征图经过转换处理后输出
        W_y = self.W(y)

        return W_y, sigm_psi_f


    def _concatenation_residual(self, x, g):
        # 获取和检测维度信息
        input_size = x.size()
        bs = input_size[0]
        assert bs == g.size(0)

        # 输入特征图转换
        theta_x = self.theta(x)
        # 获取转换后输入特征图维度
        theta_x_size = theta_x.size()

        # 门控特征图转换并且上采样到和转换后的输入特征图一样的大小
        phi_g = F.interpolate(self.phi(g), size=theta_x_size[2:], mode="trilinear", align_corners=True)
        # 将转换后的输入特征图和，门控特征图相加，经过一个激活函数
        f = F.relu(theta_x + phi_g, inplace=True)

        # 将注意力特征图转换为通道为1，然后对所有注意力分数进行Softmax归一化
        f = self.psi(f).view(bs, 1, -1)
        softmax_psi_f = torch.softmax(f, dim=2).view(bs, 1, *theta_x_size[2:])

        # 将注意力分数矩阵上采样到和输入特征图一样大小
        softmax_psi_f = F.interpolate(softmax_psi_f, size=input_size[2:], mode="trilinear", align_corners=True)
        # 将注意力分数矩阵和原输入特征图逐元素相乘
        y = softmax_psi_f.expand_as(x) * x
        # 将增强后的特征图经过转换处理后输出
        W_y = self.W(y)

        return W_y, softmax_psi_f





if __name__ == '__main__':
    model = GridAttentionGate3d(128, 256, 64, mode="concatenation", sub_sample_factor=2)

    x = torch.rand((4, 128, 80, 80, 48))
    g = torch.rand((4, 256, 40, 40, 24))

    W_y, sigm_psi_f = model(x, g)

    print(x.size())
    print(g.size())
    print(W_y.size())
    print(sigm_psi_f.size())












