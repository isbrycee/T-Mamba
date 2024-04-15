# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   yykzhjh@163.com
@DateTime :   2023/5/1 18:36
@Version  :   1.0
@License  :   (C)Copyright 2023
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.utils import *


class DICE(object):
    def __init__(self, num_classes=2, sigmoid_normalization=False, mode="extension"):
        """
        定义DICE系数评价指标计算器
        Args:
            num_classes: 类别数
            sigmoid_normalization: 对网络输出采用sigmoid归一化方法，否则采用softmax
            mode: DSC的计算方式，"standard":标准计算方式；"extension":扩展计算方式
        """
        super(DICE, self).__init__()
        # 初始化参数
        self.num_classes = num_classes
        self.mode = mode
        # 初始化sigmoid或者softmax归一化方法
        if sigmoid_normalization:
            self.normalization = nn.Sigmoid()
        else:
            self.normalization = nn.Softmax(dim=1)


    def __call__(self, input, target, no_norm_and_softmax=False):
        """
        计算DICE系数
        Args:
            input: 网络模型输出的预测图,(B, C, H, W)
            target: 标注图像,(B, H, W)

        Returns:
        """
        # 对预测图进行Sigmiod或者Sofmax归一化操作
        if no_norm_and_softmax:
            seg = input.numpy().astype(float)
            target = target.numpy().astype(float)
            return cal_dsc(seg, target)

        if input.size()[-1] == (target.size()[-1] / 4):
            input = F.interpolate(input, size=(640, 1280), mode='bilinear', align_corners=False)
        input = self.normalization(input)
        # 将预测图像进行分割
        seg = torch.argmax(input, dim=1)
        # 判断预测图和真是标签图的维度大小是否一致
        assert seg.shape == target.shape, "seg和target的维度大小不一致"
        # 转换seg和target数据类型为numpy.ndarray
        seg = seg.numpy().astype(float)
        target = target.numpy().astype(float)

        return cal_dsc(seg, target)





