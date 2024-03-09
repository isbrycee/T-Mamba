import torch
import torch.nn as nn
import math
import numpy as np

import sys
sys.path.append(r"D:\Projects\Python\3D-tooth-segmentation\PMFS-Net：Polarized Multi-scale Feature Self-attention Network For CBCT Tooth Segmentation\my-code")
from lib.utils import *


class HausdorffDistance(object):
    def __init__(self, num_classes=33, sigmoid_normalization=False):
        """
        定义豪斯多夫距离评价指标计算器

        :param num_classes: 类别数
        :param sigmoid_normalization: 对网络输出采用sigmoid归一化方法，否则采用softmax
        """
        super(HausdorffDistance, self).__init__()
        # 初始化参数
        self.num_classes = num_classes
        # 初始化sigmoid或者softmax归一化方法
        if sigmoid_normalization:
            self.normalization = nn.Sigmoid()
        else:
            self.normalization = nn.Softmax(dim=1)


    def __call__(self, input, target):
        """
        计算豪斯多夫距离评价指标

        :param input: 网络模型输出的预测图,(B, C, H, W, D)
        :param target: 标注图像,(B, H, W, D)
        :return: 各batch中各类别牙齿的豪斯多夫距离
        """
        # 对预测图进行Sigmiod或者Sofmax归一化操作
        input = self.normalization(input)

        # 将预测图像进行分割
        seg = torch.argmax(input, dim=1)
        # 判断预测图和真是标签图的维度大小是否一致
        assert seg.shape == target.shape, "seg和target的维度大小不一致"
        # 转换seg和target数据类型为整型
        seg = seg.long()
        target = target.long()

        # 将分割图和标签图都进行one-hot处理
        seg = expand_as_one_hot(seg, self.num_classes)
        target = expand_as_one_hot(target, self.num_classes)

        # 转换seg和target数据类型为布尔型
        seg = seg.bool()
        target = target.bool()

        # 判断one-hot处理后标注图和分割图的维度是否都是5维
        assert seg.dim() == target.dim() == 5, "one-hot处理后标注图和分割图的维度不是都为5维！"
        # 判断one-hot处理后标注图和分割图的尺寸是否一致
        assert seg.size() == target.size(), "one-hot处理后分割图和标注图的尺寸不一致！"

        return compute_per_channel_hd(seg, target, self.num_classes)


if __name__ == '__main__':
    pred = torch.randn((4, 33, 32, 32, 16))
    gt = torch.randint(33, (4, 32, 32, 16))

    HD_metric = HausdorffDistance(num_classes=33, c=18, sigmoid_normalization=False)
    t1 = time.time()
    HD_per_channel = HD_metric(pred, gt)
    t2 = time.time()
    print("计算耗时不用库：{:.2f}秒".format(t2 - t1))
    print(HD_per_channel)

    HD_metric = HausdorffDistance_lib(num_classes=33, sigmoid_normalization=False)
    t1 = time.time()
    HD_per_channel = HD_metric(pred, gt)
    t2 = time.time()
    print("计算耗时用库：{:.2f}秒".format(t2 - t1))
    print(HD_per_channel)




















