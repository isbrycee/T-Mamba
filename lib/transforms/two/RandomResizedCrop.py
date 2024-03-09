# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   1378453948@qq.com
@DateTime :   2023/10/15 15:58
@Version  :   1.0
@License  :   (C)Copyright 2023
"""
from __future__ import division
import torch
import math
import cv2
import random
import numpy as np
import numbers
import collections
import warnings
import PIL


INTER_MODE = {'NEAREST': cv2.INTER_NEAREST, 'BILINEAR': cv2.INTER_LINEAR, 'BICUBIC': cv2.INTER_CUBIC}
Iterable = collections.abc.Iterable


class RandomResizedCrop(object):
    """Crop the given CV Image to random size and aspect ratio.

    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.

    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation: Default: PIL.Image.BILINEAR
    """
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation='BILINEAR'):
        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (size, size)
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")

        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (CV Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        area = img.shape[0] * img.shape[1]

        for attempt in range(10):
            target_area = random.uniform(*scale) * area
            aspect_ratio = random.uniform(*ratio)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= img.shape[1] and h <= img.shape[0]:
                i = random.randint(0, img.shape[0] - h)
                j = random.randint(0, img.shape[1] - w)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = img.shape[1] / img.shape[0]
        if (in_ratio < min(ratio)):
            w = img.shape[1]
            h = int(round(w / min(ratio)))
        elif (in_ratio > max(ratio)):
            h = img.shape[0]
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = img.shape[1]
            h = img.shape[0]
        i = (img.shape[0] - h) // 2
        j = (img.shape[1] - w) // 2
        return i, j, h, w

    def __call__(self, image, label):
        """
        :param image: img (CV Image): image to be cropped and resized.
        :param label: img (CV Image): label to be cropped and resized.
        :return: CV Image: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(image, self.scale, self.ratio)
        return resized_crop(image, i, j, h, w, self.size, self.interpolation), resized_crop(label, i, j, h, w, self.size, "NEAREST")

    def __repr__(self):
        interpolate_str = self.interpolation
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
        format_string += ', ratio={0}'.format(tuple(round(r, 4) for r in self.ratio))
        format_string += ', interpolation={0})'.format(interpolate_str)
        return format_string


def _is_numpy_image(img):
    return img.ndim in {2, 3}


def crop(img, i, j, h, w):
    """Crop the given CV Image.

    Args:
        img (PIL Image): Image to be cropped.
        i (int): i in (i,j) i.e coordinates of the upper left corner.
        j (int): j in (i,j) i.e coordinates of the upper left corner.
        h (int): Height of the cropped image.
        w (int): Width of the cropped image.

    Returns:
        PIL Image: Cropped image.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be CV Image. Got {}'.format(type(img)))

    return img[i:i + h, j:j + w, ...].copy()


def resize(img, size, interpolation='BILINEAR'):
    r"""Resize the input CV Image to the given size.

    Args:
        img (CV Image): Image to be resized.
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), the output size will be matched to this. If size is an int,
            the smaller edge of the image will be matched to this number maintaing
            the aspect ratio. i.e, if height > width, then image will be rescaled to
            :math:`\left(\text{size} \times \frac{\text{height}}{\text{width}}, \text{size}\right)`
        interpolation (int, optional): Desired interpolation. Default is
            ``BILINEAR``

    Returns:
        PIL Image: Resized image.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be CV Image. Got {}'.format(type(img)))
    if not (isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)):
        raise TypeError('Got inappropriate size arg: {}'.format(size))

    interpolation = INTER_MODE[interpolation]
    if isinstance(size, int):
        h, w, _ = img.shape
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)
    else:
        oh, ow = map(int, size)
    return cv2.resize(img, (ow, oh), interpolation=interpolation)


def resized_crop(img, i, j, h, w, size, interpolation='BILINEAR'):
    """Crop the given CV Image and resize it to desired size.

    Args:
        img (CV Image): Image to be cropped.
        i (int): i in (i,j) i.e coordinates of the upper left corner
        j (int): j in (i,j) i.e coordinates of the upper left corner
        h (int): Height of the cropped image.
        w (int): Width of the cropped image.
        size (sequence or int): Desired output size. Same semantics as ``resize``.
        interpolation (int, optional): Desired interpolation. Default is
            ``BILINEAR``.
    Returns:
        CV Image: Cropped image.
    """
    assert _is_numpy_image(img), 'img should be CV Image'
    tmp_shape = img.shape
    img = crop(img, i, j, h, w)
    if img.shape[0] == 0:
        print(i, j, h, w, tmp_shape)
    img = resize(img, size, interpolation)
    return img


