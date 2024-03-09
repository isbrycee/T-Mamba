# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   1378453948@qq.com
@DateTime :   2023/10/15 16:30
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


class Cutout(object):
    """Random erase the given CV Image.

    It has been proposed in
    `Improved Regularization of Convolutional Neural Networks with Cutout`.
    `https://arxiv.org/pdf/1708.04552.pdf`


    Arguments:
        p (float): probability of the image being perspectively transformed. Default value is 0.5
        scale: range of proportion of erased area against input image.
        ratio: range of aspect ratio of erased area.
        pixel_level (bool): filling one number or not. Default value is False
    """
    def __init__(self, p=0.5, scale=(0.02, 0.4), ratio=(0.4, 1 / 0.4), value=(0, 255), pixel_level=False, inplace=False):

        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")
        if scale[0] < 0 or scale[1] > 1:
            raise ValueError("range of scale should be between 0 and 1")
        if p < 0 or p > 1:
            raise ValueError("range of random erasing probability should be between 0 and 1")
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value
        self.pixel_level = pixel_level
        self.inplace = inplace

    @staticmethod
    def get_params(img, scale, ratio):
        if type(img) == np.ndarray:
            img_h, img_w, img_c = img.shape
        else:
            img_h, img_w = img.size
            img_c = len(img.getbands())

        s = random.uniform(*scale)
        r_1 = random.uniform(*ratio)
        r_2 = random.uniform(*ratio)
        # if you img_h != img_w you may need this.
        r_1 = max(r_1, (img_h*s)/img_w)
        r_2 = min(r_2, img_h / (img_w*s))
        # r = random.uniform(*ratio)
        s = s * img_h * img_w
        w = int(math.sqrt(s / r_1))
        h = int(math.sqrt(s * r_2))
        # w = int(math.sqrt(s / r))
        # h = int(math.sqrt(s * r))
        left = random.randint(0, img_w - w)
        top = random.randint(0, img_h - h)
        
        return left, top, h, w, img_c

    def __call__(self, image, label):
        if random.random() < self.p:
            left, top, h, w, ch = self.get_params(image, self.scale, self.ratio)

            if self.pixel_level:
                c = np.random.randint(*self.value, size=(h, w, ch), dtype='uint8')
            else:
                c = random.randint(*self.value)

            label[top:top + h, left:left + w] = 0
            if type(image) == np.ndarray:
                return F.cutout(image, top, left, h, w, c, self.inplace), label
            else:
                if self.pixel_level:
                    c = PIL.Image.fromarray(c)
                image.paste(c, (left, top, left + w, top + h))
                return image, label
        return image, label

