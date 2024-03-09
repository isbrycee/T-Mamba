# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   1378453948@qq.com
@DateTime :   2024/01/01 00:33
@Version  :   1.0
@License  :   (C)Copyright 2024
"""
import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch

from torchvision import transforms


class MMOTUTester:
    """
    Tester class
    """

    def __init__(self, opt, model, metrics=None):
        self.opt = opt
        self.model = model
        self.metrics = metrics
        self.device = self.opt["device"]

        self.statistics_dict = self.init_statistics_dict()

    def inference(self, image_path):
        test_transforms = transforms.Compose([
            transforms.Resize(self.opt["resize_shape"]),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.opt["normalize_means"], std=self.opt["normalize_stds"])
        ])

        image_pil = Image.open(image_path)
        w, h = image_pil.size
        image = test_transforms(image_pil)
        dir_path, image_name = os.path.split(image_path)
        dot_pos = image_name.find(".")
        file_name = image_name[:dot_pos]
        segmentation_image_path = os.path.join(dir_path, file_name + "_segmentation" + ".jpg")

        self.model.eval()
        with torch.no_grad():
            image = torch.unsqueeze(image, dim=0)
            image = image.to(self.device)
            output = self.model(image)

        segmented_image = torch.argmax(output, dim=1).squeeze(0).to(dtype=torch.uint8).cpu().numpy()
        segmented_image = cv2.resize(segmented_image, (w, h), interpolation=cv2.INTER_AREA)
        segmented_image[segmented_image == 1] = 255
        print(segmented_image.max())
        cv2.imwrite(segmentation_image_path, segmented_image)
        print("Save segmented image to {}".format(segmentation_image_path))

    def evaluation(self, dataloader):
        self.reset_statistics_dict()
        self.model.eval()

        with torch.no_grad():
            for input_tensor, target in tqdm(dataloader, leave=True):
                input_tensor, target = input_tensor.to(self.device), target.to(self.device)
                output = self.model(input_tensor)
                self.calculate_metric_and_update_statistcs(output.cpu(), target.cpu(), len(target))

        dsc = self.statistics_dict["DSC"]["avg"] / self.statistics_dict["count"]
        class_IoU = self.statistics_dict["total_area_intersect"] / self.statistics_dict["total_area_union"]
        class_IoU = np.nan_to_num(class_IoU)
        mIoU = np.mean(class_IoU)

        print("valid_dsc:{:.6f}  valid_IoU:{:.6f}  valid_mIoU:{:.6f}".format(dsc, class_IoU[1], mIoU))

    def calculate_metric_and_update_statistcs(self, output, target, cur_batch_size):
        mask = torch.zeros(self.opt["classes"])
        unique_index = torch.unique(target).int()
        for index in unique_index:
            mask[index] = 1
        self.statistics_dict["count"] += cur_batch_size
        for i, class_name in self.opt["index_to_class_dict"].items():
            if mask[i] == 1:
                self.statistics_dict["class_count"][class_name] += cur_batch_size
        for metric_name, metric_func in self.metrics.items():
            if metric_name == "IoU":
                area_intersect, area_union, _, _ = metric_func(output, target)
                self.statistics_dict["total_area_intersect"] += area_intersect.numpy()
                self.statistics_dict["total_area_union"] += area_union.numpy()
            else:
                per_class_metric = metric_func(output, target)
                per_class_metric = per_class_metric * mask
                self.statistics_dict[metric_name]["avg"] += (torch.sum(per_class_metric) / torch.sum(mask)).item() * cur_batch_size
                for j, class_name in self.opt["index_to_class_dict"].items():
                    self.statistics_dict[metric_name][class_name] += per_class_metric[j].item() * cur_batch_size

    def init_statistics_dict(self):
        statistics_dict = {
            metric_name: {class_name: 0.0 for _, class_name in self.opt["index_to_class_dict"].items()}
            for metric_name in self.opt["metric_names"]
        }
        statistics_dict["total_area_intersect"] = np.zeros((self.opt["classes"],))
        statistics_dict["total_area_union"] = np.zeros((self.opt["classes"],))
        for metric_name in self.opt["metric_names"]:
            statistics_dict[metric_name]["avg"] = 0.0
        statistics_dict["class_count"] = {class_name: 0 for _, class_name in self.opt["index_to_class_dict"].items()}
        statistics_dict["count"] = 0

        return statistics_dict

    def reset_statistics_dict(self):
        self.statistics_dict["count"] = 0
        self.statistics_dict["total_area_intersect"] = np.zeros((self.opt["classes"],))
        self.statistics_dict["total_area_union"] = np.zeros((self.opt["classes"],))
        for _, class_name in self.opt["index_to_class_dict"].items():
            self.statistics_dict["class_count"][class_name] = 0
        for metric_name in self.opt["metric_names"]:
            self.statistics_dict[metric_name]["avg"] = 0.0
            for _, class_name in self.opt["index_to_class_dict"].items():
                self.statistics_dict[metric_name][class_name] = 0.0

    def load(self):
        pretrain_state_dict = torch.load(self.opt["pretrain"], map_location=lambda storage, loc: storage.cuda(self.device))
        model_state_dict = self.model.state_dict()
        load_count = 0
        for param_name in model_state_dict.keys():
            if (param_name in pretrain_state_dict) and (model_state_dict[param_name].size() == pretrain_state_dict[param_name].size()):
                model_state_dict[param_name].copy_(pretrain_state_dict[param_name])
                load_count += 1
        self.model.load_state_dict(model_state_dict, strict=True)
        print("{:.2f}% of model parameters successfully loaded with training weights".format(100 * load_count / len(model_state_dict)))
