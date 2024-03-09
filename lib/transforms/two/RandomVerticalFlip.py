# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   1378453948@qq.com
@DateTime :   2023/10/15 16:27
@Version  :   1.0
@License  :   (C)Copyright 2023
"""
from __future__ import division
import torch
import math
import random
import numpy as np
import numbers
import collections
import warnings
import PIL

from torchtoolbox.transform import functional as F


class RandomVerticalFlip(object):
    """Vertically flip the given CV Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, label):
        """
        :param image: img (CV Image): image to be flipped.
        :param label: img (CV Image): label to be flipped.
        :return: img (CV Image): image and label to be flipped.
        """
        if random.random() < self.p:
            return F.vflip(image), F.vflip(label)
        return image, label

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

