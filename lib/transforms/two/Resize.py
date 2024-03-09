# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   1378453948@qq.com
@DateTime :   2023/10/15 16:44
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

Iterable = collections.abc.Iterable


class Resize(object):
    """Resize the input CV Image to the given size.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``BILINEAR``
    """
    def __init__(self, size, interpolation='BILINEAR'):
        assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, image, label):
        """
        :param image: (CV Image): image to be scaled.
        :param label: (CV Image): label to be scaled.
        :return: CV Image: Rescaled image and label
        """
        return F.resize(image, self.size, self.interpolation), F.resize(label, self.size, "NEAREST")

    def __repr__(self):
        interpolate_str = self.interpolation
        return self.__class__.__name__ + \
               '(size={0}, interpolation={1})'.format(self.size, interpolate_str)

