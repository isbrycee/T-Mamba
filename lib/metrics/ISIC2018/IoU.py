# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   yykzhjh@163.com
@DateTime :   2023/6/13 2:54
@Version  :   1.0
@License  :   (C)Copyright 2023
"""
import math
import torch
import torch.nn as nn
import numpy as np

import sys
sys.path.append(r"D:\Projects\Python\3D-tooth-segmentation\PMFS-Net：Polarized Multi-scale Feature Self-attention Network For CBCT Tooth Segmentation\my-2d")
from lib.utils import *



class IoU(object):
    def __init__(self, num_classes=33, sigmoid_normalization=False):
        """
        定义IoU评价指标计算器

        :param num_classes: 类别数
        :param sigmoid_normalization: 对网络输出采用sigmoid归一化方法，否则采用softmax
        """
        super(IoU, self).__init__()
        # 初始化参数
        self.num_classes = num_classes
        # 初始化sigmoid或者softmax归一化方法
        if sigmoid_normalization:
            self.normalization = nn.Sigmoid()
        else:
            self.normalization = nn.Softmax(dim=1)


    def __call__(self, input, target, no_norm_and_softmax=False):
        """
        IoU

        :param input: 网络模型输出的预测图,(B, C, H, W)
        :param target: 标注图像,(B, H, W)
        :return:
        """
        if no_norm_and_softmax:
            seg = input.type(torch.float32)
            target = target.type(torch.float32)
            return intersect_and_union(seg, target, self.num_classes, reduce_zero_label=False)

        # 对预测图进行Sigmiod或者Sofmax归一化操作
        if input.size()[-1] == (target.size()[-1] / 4):
            import torch.nn.functional as F
            input = F.interpolate(input, size=(640, 1280), mode='bilinear', align_corners=False)
        input = self.normalization(input)

        # 将预测图像进行分割
        seg = torch.argmax(input, dim=1)
        # 判断预测图和真是标签图的维度大小是否一致
        assert seg.shape == target.shape, "seg和target的维度大小不一致"
        # 转换seg和target数据类型为整型
        seg = seg.type(torch.float32)
        target = target.type(torch.float32)

        return intersect_and_union(seg, target, self.num_classes, reduce_zero_label=False)




if __name__ == '__main__':
    random.seed(123)  # 为python设置随机种子
    os.environ['PYTHONHASHSEED'] = str(123)
    np.random.seed(123)  # 为numpy设置随机种子
    torch.manual_seed(123)  # 为CPU设置随机种子

    pred = torch.randn((4, 2, 32, 32))
    gt = torch.randint(2, (4, 32, 32))

    IoU_metric = IoU(num_classes=2)

    intersect, union, _, _ = IoU_metric(pred, gt)

    print((intersect[1].to(torch.float64) / (union[1].to(torch.float64) + 1e-6)).item())




















