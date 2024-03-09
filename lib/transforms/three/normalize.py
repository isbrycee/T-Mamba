# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   yykzhjh@163.com
@DateTime :   2022/11/8 17:03
@Version  :   1.0
@License  :   (C)Copyright 2022
"""
import torch



class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std



    def __call__(self, img_tensor, label=None):
        """
        Args:
            img_tensor: Image to be normalized
            label: Label segmentation map to be normalized

        Returns:
        """
        # 标准化图像数据
        img_tensor = (img_tensor - self.mean) / self.std
        return img_tensor, label




