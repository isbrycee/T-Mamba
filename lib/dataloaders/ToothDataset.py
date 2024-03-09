# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   yykzhjh@163.com
@DateTime :   2023/4/19 16:28
@Version  :   1.0
@License  :   (C)Copyright 2023
"""
import os
import glob
import numpy as np

from torch.utils.data import Dataset

import lib.transforms.three as transforms
import lib.utils as utils


class ToothDataset(Dataset):
    """
    load 3D CBCT Tooth Dataset
    """

    def __init__(self, opt, mode):
        """
        Args:
            opt: params dictionary
            mode: train/valid
        """
        self.opt = opt
        self.mode = mode
        self.root = opt["dataset_path"]
        self.train_path = os.path.join(self.root, "train")
        self.val_path = os.path.join(self.root, "valid")

        if self.mode == 'train':
            self.augmentations = [
                opt["open_elastic_transform"], opt["open_gaussian_noise"], opt["open_random_flip"],
                opt["open_random_rescale"], opt["open_random_rotate"], opt["open_random_shift"]]
            self.sub_volume_root_dir = os.path.join(self.root, "sub_volumes")
            if not os.path.exists(self.sub_volume_root_dir):
                os.makedirs(self.sub_volume_root_dir)
            self.sub_volume_path = os.path.join(self.sub_volume_root_dir,
                                                "-".join([str(item) for item in opt["crop_size"]])
                                                + "_" + str(opt["samples_train"]) + ".npz")
            self.selected_images = []
            self.selected_position = []
            all_augments = [
                 transforms.ElasticTransform(alpha=opt["elastic_transform_alpha"],
                                             sigma=opt["elastic_transform_sigma"]),
                 transforms.GaussianNoise(mean=opt["gaussian_noise_mean"],
                                          std=opt["gaussian_noise_std"]),
                 transforms.RandomFlip(),
                 transforms.RandomRescale(min_percentage=opt["random_rescale_min_percentage"],
                                          max_percentage=opt["random_rescale_max_percentage"]),
                 transforms.RandomRotation(min_angle=opt["random_rotate_min_angle"],
                                           max_angle=opt["random_rotate_max_angle"]),
                 transforms.RandomShift(max_percentage=opt["random_shift_max_percentage"])
            ]
            practice_augments = [all_augments[i] for i, is_open in enumerate(self.augmentations) if is_open]
            if opt["augmentation_method"] == "Choice":
                self.train_transforms = transforms.ComposeTransforms([
                    transforms.ClipAndShift(opt["clip_lower_bound"], opt["clip_upper_bound"]),
                    transforms.RandomAugmentChoice(practice_augments, p=opt["augmentation_probability"]),
                    transforms.ToTensor(opt["clip_lower_bound"], opt["clip_upper_bound"]),
                    transforms.Normalize(opt["normalize_mean"], opt["normalize_std"])
                ])
            elif opt["augmentation_method"] == "Compose":
                self.train_transforms = transforms.ComposeTransforms([
                    transforms.ClipAndShift(opt["clip_lower_bound"], opt["clip_upper_bound"]),
                    transforms.ComposeAugments(practice_augments, p=opt["augmentation_probability"]),
                    transforms.ToTensor(opt["clip_lower_bound"], opt["clip_upper_bound"]),
                    transforms.Normalize(opt["normalize_mean"], opt["normalize_std"])
                ])

            if (not opt["create_data"]) and os.path.isfile(self.sub_volume_path):
                sub_volume_dict = np.load(self.sub_volume_path)
                self.selected_images = [tuple(image) for image in sub_volume_dict["selected_images"]]
                self.selected_position = [tuple(crop_point) for crop_point in sub_volume_dict["selected_position"]]
            else:
                images_path_list = sorted(glob.glob(os.path.join(self.train_path, "images", "*.nii.gz")))
                labels_path_list = sorted(glob.glob(os.path.join(self.train_path, "labels", "*.nii.gz")))

                self.selected_images, self.selected_position = utils.create_sub_volumes(images_path_list, labels_path_list, opt)

                np.savez(self.sub_volume_path, selected_images=self.selected_images, selected_position=self.selected_position)

        elif self.mode == 'valid':
            self.val_transforms = transforms.ComposeTransforms([
                transforms.ClipAndShift(opt["clip_lower_bound"], opt["clip_upper_bound"]),
                transforms.ToTensor(opt["clip_lower_bound"], opt["clip_upper_bound"]),
                transforms.Normalize(opt["normalize_mean"], opt["normalize_std"])
            ])

            images_path_list = sorted(glob.glob(os.path.join(self.val_path, "images", "*.nii.gz")))
            labels_path_list = sorted(glob.glob(os.path.join(self.val_path, "labels", "*.nii.gz")))

            self.selected_images = list(zip(images_path_list, labels_path_list))


    def __len__(self):
        return len(self.selected_images)

    def __getitem__(self, index):
        image_path, label_path = self.selected_images[index]
        image_np = utils.load_image_or_label(image_path, self.opt["resample_spacing"], type="image")
        label_np = utils.load_image_or_label(label_path, self.opt["resample_spacing"], type="label")

        if self.mode == 'train':
            crop_point = self.selected_position[index]
            crop_image_np = utils.crop_img(image_np, self.opt["crop_size"], crop_point)
            crop_label_np = utils.crop_img(label_np, self.opt["crop_size"], crop_point)
            transform_image, transform_label = self.train_transforms(crop_image_np, crop_label_np)
            return transform_image.unsqueeze(0), transform_label

        else:
            transform_image, transform_label = self.val_transforms(image_np, label_np)
            return transform_image.unsqueeze(0), transform_label
