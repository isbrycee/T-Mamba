# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   1378453948@qq.com
@DateTime :   2023/10/29 01:02
@Version  :   1.0
@License  :   (C)Copyright 2023
"""
import os
import glob

import cv2
import numpy as np
from PIL import Image

from torch.utils.data import Dataset

import lib.utils as utils
import lib.transforms.two as my_transforms


class ISIC2018Dataset(Dataset):
    """
    load ISIC 2018 dataset
    """

    def __init__(self, opt, mode):
        """
        initialize ISIC 2018 dataset
        :param opt: params dict
        :param mode: train/valid
        """
        super(ISIC2018Dataset, self).__init__()
        self.opt = opt
        self.mode = mode
        self.root = opt["dataset_path"]
        self.train_dir = os.path.join(self.root, "train")
        self.valid_dir = os.path.join(self.root, "test")
        self.transforms_dict = {
            "train": my_transforms.Compose([
                my_transforms.RandomResizedCrop(self.opt["resize_shape"], scale=(0.4, 1.0), ratio=(3. / 4., 4. / 3.), interpolation='BILINEAR'),
                my_transforms.ColorJitter(brightness=self.opt["color_jitter"], contrast=self.opt["color_jitter"], saturation=self.opt["color_jitter"], hue=0),
                my_transforms.RandomGaussianNoise(p=self.opt["augmentation_p"]),
                my_transforms.RandomHorizontalFlip(p=self.opt["augmentation_p"]),
                my_transforms.RandomVerticalFlip(p=self.opt["augmentation_p"]),
                my_transforms.RandomRotation(self.opt["random_rotation_angle"]),
                my_transforms.Cutout(p=self.opt["augmentation_p"], value=(0, 0)),
                my_transforms.ToTensor(),
                my_transforms.Normalize(mean=self.opt["normalize_means"], std=self.opt["normalize_stds"])
            ]),
            "valid": my_transforms.Compose([
                my_transforms.Resize(self.opt["resize_shape"]),
                my_transforms.ToTensor(),
                my_transforms.Normalize(mean=self.opt["normalize_means"], std=self.opt["normalize_stds"])
            ])
        }

        if mode == "train":
            self.images_list = sorted(glob.glob(os.path.join(self.train_dir, "images", "*.jpg")))
            self.labels_list = sorted(glob.glob(os.path.join(self.train_dir, "annotations", "*.png")))
        else:
            self.images_list = sorted(glob.glob(os.path.join(self.valid_dir, "images", "*.jpg")))
            self.labels_list = sorted(glob.glob(os.path.join(self.valid_dir, "annotations", "*.png")))

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, index):
        image = cv2.imread(self.images_list[index], -1)
        label = cv2.imread(self.labels_list[index], -1)
        label[label == 255] = 1
        image, label = self.transforms_dict[self.mode](image, label)
        return image, label
