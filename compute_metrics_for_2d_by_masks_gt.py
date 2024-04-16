# -*- encoding: utf-8 -*-
"""
@author   :   Bryce
@Contact  :   isjinghao@gmail.com
@DateTime :   2024/04/16 10:21
@Version  :   1.0
@License  :   (C)Copyright 2024
"""
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
import torch
import argparse
from lib import metrics
from torchvision import transforms


class ComputeMetricfor2D:

    def __init__(self, opt, metrics=None):
        self.opt = opt
        self.metrics = metrics
        self.statistics_dict = self.init_statistics_dict()
        self.transform = transforms.Compose([
            transforms.ToTensor(),  
        ])

    def compute(self, gt_dir, pred_mask_dir):
        self.reset_statistics_dict()
        gt_pred_map_dict = {}
        gt_list = os.listdir(gt_dir)
        for gt_path in gt_list:
            img_name = gt_path.split('/')[-1]
            gt_pred_map_dict[gt_path] = os.path.join(pred_mask_dir, img_name)

        for gt_path in tqdm(gt_list):
            import torch.nn.functional as F
            gt = self.transform(Image.open(os.path.join(gt_dir, gt_path)))
            # gt = torch.unsqueeze(gt, dim=0)
            # gt = F.interpolate(gt, size=(640, 1280), mode='bilinear', align_corners=False)

            pred = self.transform(Image.open(gt_pred_map_dict[gt_path]))
            # pred = torch.unsqueeze(pred, dim=0)
            # pred = F.interpolate(pred, size=(640, 1280), mode='bilinear', align_corners=False)

            self.calculate_metric_and_update_statistcs(pred.cpu(), gt.cpu(), len(gt))

        class_IoU = self.statistics_dict["total_area_intersect"] / self.statistics_dict["total_area_union"]
        class_IoU = np.nan_to_num(class_IoU)
        dsc = self.statistics_dict["DSC_sum"] / self.statistics_dict["count"]
        JI = self.statistics_dict["JI_sum"] / self.statistics_dict["count"]
        ACC = self.statistics_dict["ACC_sum"] / self.statistics_dict["count"]

        print("valid_DSC:{:.6f}  valid_IoU:{:.6f}  valid_ACC:{:.6f}  valid_JI:{:.6f}".format(dsc, class_IoU[1], ACC, JI))


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
                area_intersect, area_union, _, _ = metric_func(output, target, no_norm_and_softmax=True)
                self.statistics_dict["total_area_intersect"] += area_intersect.numpy()
                self.statistics_dict["total_area_union"] += area_union.numpy()
            elif metric_name == "ACC":
                batch_mean_ACC = metric_func(output, target, no_norm_and_softmax=True)
                self.statistics_dict["ACC_sum"] += batch_mean_ACC * cur_batch_size
            elif metric_name == "JI":
                batch_mean_JI = metric_func(output, target, no_norm_and_softmax=True)
                self.statistics_dict["JI_sum"] += batch_mean_JI * cur_batch_size
            elif metric_name == "DSC":
                batch_mean_DSC = metric_func(output, target, no_norm_and_softmax=True)
                self.statistics_dict["DSC_sum"] += batch_mean_DSC * cur_batch_size
            else:
                per_class_metric = metric_func(output, target, no_norm_and_softmax=True)
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
        statistics_dict["JI_sum"] = 0.0
        statistics_dict["ACC_sum"] = 0.0
        statistics_dict["DSC_sum"] = 0.0
        for metric_name in self.opt["metric_names"]:
            statistics_dict[metric_name]["avg"] = 0.0
        statistics_dict["class_count"] = {class_name: 0 for _, class_name in self.opt["index_to_class_dict"].items()}
        statistics_dict["count"] = 0

        return statistics_dict

    def reset_statistics_dict(self):
        self.statistics_dict["count"] = 0
        self.statistics_dict["total_area_intersect"] = np.zeros((self.opt["classes"],))
        self.statistics_dict["total_area_union"] = np.zeros((self.opt["classes"],))
        self.statistics_dict["JI_sum"] = 0.0
        self.statistics_dict["ACC_sum"] = 0.0
        self.statistics_dict["DSC_sum"] = 0.0
        for _, class_name in self.opt["index_to_class_dict"].items():
            self.statistics_dict["class_count"][class_name] = 0
        for metric_name in self.opt["metric_names"]:
            self.statistics_dict[metric_name]["avg"] = 0.0
            for _, class_name in self.opt["index_to_class_dict"].items():
                self.statistics_dict[metric_name][class_name] = 0.0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_dir", type=str, default="/root/paddlejob/workspace/env_run/output/haojing08/gem/MaskDINO-main-Gem_/datasets/Tooth2D-X-Ray-6k/test/annotations", help="dataset name")
    parser.add_argument("--mask_dir", type=str, default="/root/paddlejob/workspace/env_run/output/haojing08/gem/MaskDINO-main-Gem_/datasets/Tooth2D-X-Ray-6k/test_mask_gem", help="model name")
    args = parser.parse_args()
    return args

def main():
    # analyse console arguments
    args = parse_args()

    # initialize the metrics
    params = {
        "dataset_name": "Tooth2D-X-Ray-6k",
        "classes": 2,
        "index_to_class_dict":
        {
            0: "background",
            1: "foreground"
        },
        # ————————————————————————————————————————————    Loss And Metric     ———————————————————————————————————————————————————————
        "metric_names": ["DSC", "IoU", "JI", "ACC"],
        "loss_function_name": "DiceLoss",
        "class_weight": [0.029, 1-0.029],
        "sigmoid_normalization": False,
        "dice_loss_mode": "extension",
        "dice_mode": "standard",
    }

    params["gt_dir"] = args.gt_dir
    params["mask_dir"] = args.mask_dir

    metric = metrics.get_metric(params)

    # initialize the tester
    metricComputer = ComputeMetricfor2D(params, metric)

    # evaluate valid set
    metricComputer.compute(params["gt_dir"], params["mask_dir"])


if __name__ == '__main__':
    main()