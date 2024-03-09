# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   yykzhjh@163.com
@DateTime :   2023/4/21 14:57
@Version  :   1.0
@License  :   (C)Copyright 2023
"""
import torch
import numpy as np



class ClipAndShift(object):
    def __init__(self, lower_bound, upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound


    def __call__(self, img_numpy, label):
        img_numpy[img_numpy < self.lower_bound] = self.lower_bound
        img_numpy[img_numpy > self.upper_bound] = self.upper_bound

        img_numpy -= self.lower_bound

        return img_numpy, label



