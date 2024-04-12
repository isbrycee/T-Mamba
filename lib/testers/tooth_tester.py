# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   1378453948@qq.com
@DateTime :   2024/01/01 00:32
@Version  :   1.0
@License  :   (C)Copyright 2024
"""
import os
import math
import numpy as np
from tqdm import tqdm

import torch

from lib import utils
import lib.transforms.three as transforms


class ToothTester:
    """
    Tester class
    """

    def __init__(self, opt, model, metrics=None):
        self.opt = opt
        self.model = model
        self.metrics = metrics
        self.device = self.opt["device"]

        self.class_names = list(self.opt["index_to_class_dict"].values())
        self.statistics_dict = self.init_statistics_dict()

    def inference(self, image_path):
        test_transforms = transforms.ComposeTransforms([
            transforms.ClipAndShift(self.opt["clip_lower_bound"], self.opt["clip_upper_bound"]),
            transforms.ToTensor(self.opt["clip_lower_bound"], self.opt["clip_upper_bound"]),
            transforms.Normalize(self.opt["normalize_mean"], self.opt["normalize_std"])
        ])

        image_np = utils.load_image_or_label(image_path, self.opt["resample_spacing"], type="image")
        image, _ = test_transforms(image_np, None)
        dir_path, image_name = os.path.split(image_path)
        dot_pos = image_name.find(".")
        file_name = image_name[:dot_pos]
        segmentation_image_path = os.path.join(dir_path, file_name + "_segmentation" + ".npy")

        self.model.eval()
        with torch.no_grad():
            image = torch.FloatTensor(image.numpy()).unsqueeze(0).unsqueeze(0)
            image = image.to(self.device)
            output = self.split_test(image)

        segmented_image = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
        # np.save(segmentation_image_path, segmented_image)
        print("Save segmented image to {}".format(segmentation_image_path))


    def evaluation(self, dataloader):
        self.reset_statistics_dict()
        self.model.eval()

        with torch.no_grad():
            for image, label in tqdm(dataloader, leave=True):
                image = image.to(self.device)
                label = label.to(self.device)
                output = self.split_test(image)
                per_class_metrics = []
                for metric in self.metrics:
                    per_class_metrics.append(metric(output.cpu(), label.cpu()))
                self.update_statistics_dict(per_class_metrics, label, label.size(0))

        self.display_statistics_dict()

    def split_test(self, image):
        ori_shape = image.size()[2:]
        output = torch.zeros((image.size()[0], self.opt["classes"], *ori_shape), device=self.device)
        slice_shape = self.opt["crop_size"]
        stride = self.opt["crop_stride"]
        total_slice_num = 1
        for i in range(3):
            total_slice_num *= math.ceil((ori_shape[i] - slice_shape[i]) / stride[i]) + 1

        with tqdm(total=total_slice_num, leave=False) as bar:
            for shape0_start in range(0, ori_shape[0], stride[0]):
                shape0_end = shape0_start + slice_shape[0]
                start0 = shape0_start
                end0 = shape0_end
                if shape0_end >= ori_shape[0]:
                    end0 = ori_shape[0]
                    start0 = end0 - slice_shape[0]

                for shape1_start in range(0, ori_shape[1], stride[1]):
                    shape1_end = shape1_start + slice_shape[1]
                    start1 = shape1_start
                    end1 = shape1_end
                    if shape1_end >= ori_shape[1]:
                        end1 = ori_shape[1]
                        start1 = end1 - slice_shape[1]

                    for shape2_start in range(0, ori_shape[2], stride[2]):
                        shape2_end = shape2_start + slice_shape[2]
                        start2 = shape2_start
                        end2 = shape2_end
                        if shape2_end >= ori_shape[2]:
                            end2 = ori_shape[2]
                            start2 = end2 - slice_shape[2]

                        slice_tensor = image[:, :, start0:end0, start1:end1, start2:end2]
                        slice_predict = self.model(slice_tensor.to(self.device))
                        output[:, :, start0:end0, start1:end1, start2:end2] += slice_predict
                        bar.update(1)

                        if shape2_end >= ori_shape[2]:
                            break

                    if shape1_end >= ori_shape[1]:
                        break

                if shape0_end >= ori_shape[0]:
                    break

        return output

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

    def init_statistics_dict(self):
        statistics_dict = {
            metric_name: {class_name: 0.0 for class_name in self.class_names}
            for metric_name in self.opt["metric_names"]
        }
        for metric_name in self.opt["metric_names"]:
            statistics_dict[metric_name]["avg"] = 0.0
        statistics_dict["class_count"] = {class_name: 0 for class_name in self.class_names}
        statistics_dict["count"] = 0

        return statistics_dict

    def reset_statistics_dict(self):
        self.statistics_dict["count"] = 0
        for class_name in self.class_names:
            self.statistics_dict["class_count"][class_name] = 0
        for metric_name in self.opt["metric_names"]:
            self.statistics_dict[metric_name]["avg"] = 0.0
            for class_name in self.class_names:
                self.statistics_dict[metric_name][class_name] = 0.0

    def update_statistics_dict(self, per_class_metrics, target, cur_batch_size):
        mask = torch.zeros(self.opt["classes"])
        unique_index = torch.unique(target).int()
        for index in unique_index:
            mask[index] = 1
        self.statistics_dict["count"] += cur_batch_size
        for i, class_name in enumerate(self.class_names):
            if mask[i] == 1:
                self.statistics_dict["class_count"][class_name] += cur_batch_size
        for i, metric_name in enumerate(self.opt["metric_names"]):
            per_class_metric = per_class_metrics[i]
            per_class_metric = per_class_metric * mask
            self.statistics_dict[metric_name]["avg"] += (torch.sum(per_class_metric) / torch.sum(
                mask)).item() * cur_batch_size
            for j, class_name in enumerate(self.class_names):
                self.statistics_dict[metric_name][class_name] += per_class_metric[j].item() * cur_batch_size

    def display_statistics_dict(self):
        print_info = ""
        print_info += " " * 12
        for metric_name in self.opt["metric_names"]:
            print_info += "{:^12}".format(metric_name)
        print_info += '\n'
        for class_name in self.class_names:
            print_info += "{:<12}".format(class_name)
            for metric_name in self.opt["metric_names"]:
                value = 0
                if self.statistics_dict["class_count"][class_name] != 0:
                    value = self.statistics_dict[metric_name][class_name] / self.statistics_dict["class_count"][class_name]
                print_info += "{:^12.6f}".format(value)
            print_info += '\n'
        print_info += "{:<12}".format("average")
        for metric_name in self.opt["metric_names"]:
            value = 0
            if self.statistics_dict["count"] != 0:
                value = self.statistics_dict[metric_name]["avg"] / self.statistics_dict["count"]
            print_info += "{:^12.6f}".format(value)
        print(print_info)
