import numpy as np
import torch
import os
import json
import re
import scipy
from PIL import Image
import SimpleITK as sitk
from scipy import ndimage
from collections import Counter
from nibabel.viewers import OrthoSlicer3D
import matplotlib.pyplot as plt
import lib.utils as utils


def create_sub_volumes(images_path_list, labels_path_list, opt):
    image_num = len(images_path_list)
    assert image_num != 0, "original dataset is empty"
    assert len(images_path_list) == len(labels_path_list), "The number of images and labels in the dataset is not equal"

    selected_images = []
    selected_position = []

    for i in range(opt["samples_train"]):
        print("id:", i)
        random_index = np.random.randint(image_num)
        print(labels_path_list[random_index])
        label_np = load_image_or_label(labels_path_list[random_index], opt["resample_spacing"], type="label")

        cnt_loop = 0
        while True:
            if label_np.shape[0]<160:
                break
            cnt_loop += 1
            crop_point = find_random_crop_dim(label_np.shape, opt["crop_size"])
            if find_non_zero_labels_mask(label_np, opt["crop_threshold"], opt["crop_size"], crop_point):
                selected_images.append((images_path_list[random_index], labels_path_list[random_index]))
                selected_position.append(crop_point)
                print("loop cnt:", cnt_loop, '\n')
                break

    return selected_images, selected_position


def find_random_crop_dim(full_vol_dim, crop_size):
    assert full_vol_dim[0] >= crop_size[0], "crop size is too big"
    assert full_vol_dim[1] >= crop_size[1], "crop size is too big"
    assert full_vol_dim[2] >= crop_size[2], "crop size is too big"

    if full_vol_dim[0] == crop_size[0]:
        slices = 0
    else:
        slices = np.random.randint(full_vol_dim[0] - crop_size[0])

    if full_vol_dim[1] == crop_size[1]:
        w_crop = 0
    else:
        w_crop = np.random.randint(full_vol_dim[1] - crop_size[1])

    if full_vol_dim[2] == crop_size[2]:
        h_crop = 0
    else:
        h_crop = np.random.randint(full_vol_dim[2] - crop_size[2])

    return (slices, w_crop, h_crop)


def find_non_zero_labels_mask(label_map, th_percent, crop_size, crop_point):
    segmentation_map = label_map.copy()
    d1, d2, d3 = segmentation_map.shape
    segmentation_map[segmentation_map > 0] = 1
    total_voxel_labels = segmentation_map.sum()

    cropped_segm_map = crop_img(segmentation_map, crop_size, crop_point)
    crop_voxel_labels = cropped_segm_map.sum()

    label_percentage = crop_voxel_labels / total_voxel_labels
    # print(label_percentage,total_voxel_labels,crop_voxel_labels)
    if label_percentage >= th_percent:
        return True
    else:
        return False


def load_image_or_label(path, resample_spacing, type=None):
    if type == "label":
        img_np, spacing = load_label(path)
        # print(img_np.shape, spacing)
    else:
        img_np, spacing = load_image(path)

    order = 0 if type == "label" else 3
    img_np = resample_image_spacing(img_np, spacing, resample_spacing, order)

    # if type == "label":
    #     print(img_np.shape)

    # img_tensor = torch.from_numpy(img_np)

    return img_np


def load_nii_file(file_path):
    NiiImage = sitk.ReadImage(file_path)
    image_numpy = sitk.GetArrayFromImage(NiiImage)
    image_numpy = image_numpy.transpose(2, 1, 0)
    spacing = NiiImage.GetSpacing()

    return image_numpy, spacing


def load_label(path):
    data, spacing = load_nii_file(path)
    # print(data.min(), data.max())

    data[data > 0] = 1

    data = data.astype("uint8")

    # print(np.sum(data > 0) / data.size)

    # OrthoSlicer3D(data).show()

    return data, spacing


def load_image(path):
    data, spacing = load_nii_file(path)
    data = data.astype("float32")

    return data, spacing


def resample_image_spacing(data, old_spacing, new_spacing, order):
    scale_list = [old / new_spacing[i] for i, old in enumerate(old_spacing)]
    return scipy.ndimage.interpolation.zoom(data, scale_list, order=order)


def crop_img(img_np, crop_size, crop_point):
    if crop_size[0] == 0:
        return img_np
    slices_crop, w_crop, h_crop = crop_point
    dim1, dim2, dim3 = crop_size
    inp_img_dim = img_np.ndim
    assert inp_img_dim >= 3
    if img_np.ndim == 3:
        full_dim1, full_dim2, full_dim3 = img_np.shape
    elif img_np.ndim == 4:
        _, full_dim1, full_dim2, full_dim3 = img_np.shape
        img_np = img_np[0, ...]

    if full_dim1 == dim1:
        img_np = img_np[:, w_crop:w_crop + dim2, h_crop:h_crop + dim3]
    elif full_dim2 == dim2:
        img_np = img_np[slices_crop:slices_crop + dim1, :, h_crop:h_crop + dim3]
    elif full_dim3 == dim3:
        img_np = img_np[slices_crop:slices_crop + dim1, w_crop:w_crop + dim2, :]
    else:
        img_np = img_np[slices_crop:slices_crop + dim1, w_crop:w_crop + dim2, h_crop:h_crop + dim3]

    if inp_img_dim == 4:
        return img_np.unsqueeze(0)

    return img_np














if __name__ == '__main__':
    load_label(r"./datasets/src_10/train/labels/12_2.nrrd")









