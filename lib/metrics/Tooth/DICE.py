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

from lib.utils import *


class DICE(object):
    def __init__(self, num_classes=33, sigmoid_normalization=False, mode="extension"):
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


    def __call__(self, input, target):
        """
        计算DICE系数
        Args:
            input: 网络模型输出的预测图,(B, C, H, W, D)
            target: 标注图像,(B, H, W, D)

        Returns:
        """
        # ont-hot处理，将标注图在axis=1维度上扩张，该维度大小等于预测图的通道C大小，维度上每一个索引依次对应一个类别,(B, C, H, W, D)
        target = expand_as_one_hot(target.long(), self.num_classes)

        # 判断one-hot处理后标注图和预测图的维度是否都是5维
        assert input.dim() == target.dim() == 5, "one-hot处理后标注图和预测图的维度不是都为5维！"
        # 判断one-hot处理后标注图和预测图的尺寸是否一致
        assert input.size() == target.size(), "one-hot处理后预测图和标注图的尺寸不一致！"

        # 对预测图进行Sigmiod或者Sofmax归一化操作
        input = self.normalization(input)

        return compute_per_channel_dice(input, target, epsilon=1e-6, mode=self.mode)





