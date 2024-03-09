# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   1378453948@qq.com
@DateTime :   2023/10/15 16:16
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



class RandomGaussianNoise(object):
    """Applying gaussian noise on the given CV Image randomly with a given probability.
        Args:
            p (float): probability of the image being noised. Default value is 0.5
            mean (float): Gaussian distribution mean if not fixed_distribution it will random in [0, mean]
            std (float): Gaussian distribution std if not fixed_distribution it will random in [0, std]
            fixed_distribution (bool): whether use a fixed distribution
        """
    def __init__(self, p=0.5, mean=0, std=0.1, fixed_distribution=True):
        assert isinstance(mean, numbers.Number) and mean >= 0, 'mean should be a positive value'
        assert isinstance(std, numbers.Number) and std >= 0, 'std should be a positive value'
        assert isinstance(p, numbers.Number) and p >= 0, 'p should be a positive value'
        self.p = p
        self.mean = mean
        self.std = std
        self.fixed_distribution = fixed_distribution

    @staticmethod
    def get_params(mean, std):
        """Get parameters for gaussian noise
        Returns:
            sequence: params to be passed to the affine transformation
        """
        mean = random.uniform(0, mean)
        std = random.uniform(0, std)

        return mean, std

    def __call__(self, image, label):
        """
        :param image: img (np.ndarray): image to be noised.
        :param label: img (np.ndarray): label to be noised.
        :return:
        """
        if random.random() < self.p:
            if self.fixed_distribution:
                mean, std = self.mean, self.std
            else:
                mean, std = self.get_params(self.mean, self.std)
            return F.gaussian_noise(image, mean=mean, std=std), label
        return image, label

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

