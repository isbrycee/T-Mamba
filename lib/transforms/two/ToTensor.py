# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   1378453948@qq.com
@DateTime :   2023/10/15 16:37
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



class ToTensor(object):
    """Convert a ``CV Image`` or ``numpy.ndarray`` to tensor.

    Converts a CV Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    if the CV Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)
    or if the numpy.ndarray has dtype = np.uint8

    In the other cases, tensors are returned without scaling.
    """
    def __call__(self, image, label):
        """
        :param image: (CV Image or numpy.ndarray): image to be converted to tensor.
        :param label: (CV Image or numpy.ndarray): label to be converted to tensor.
        :return: Tensor: Converted image and label
        """
        image = torch.from_numpy(image.transpose((2, 0, 1))).float().div(255)
        label = torch.from_numpy(label).float()
        return image, label

    def __repr__(self):
        return self.__class__.__name__ + '()'

