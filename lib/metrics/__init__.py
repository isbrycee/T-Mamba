# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   1378453948@qq.com
@DateTime :   2023/12/30 16:57
@Version  :   1.0
@License  :   (C)Copyright 2023
"""
from lib.metrics import Tooth
from lib.metrics import MMOTU
from lib.metrics import ISIC2018


def get_metric(opt):
    if opt["dataset_name"] == "3D-CBCT-Tooth":
        metrics = []
        for metric_name in opt["metric_names"]:
            if metric_name == "DSC":
                metrics.append(Tooth.DICE(num_classes=opt["classes"], sigmoid_normalization=opt["sigmoid_normalization"], mode=opt["dice_mode"]))

            elif metric_name == "ASSD":
                metrics.append(Tooth.AverageSymmetricSurfaceDistance(num_classes=opt["classes"], sigmoid_normalization=opt["sigmoid_normalization"]))

            elif metric_name == "HD":
                metrics.append(Tooth.HausdorffDistance(num_classes=opt["classes"], sigmoid_normalization=opt["sigmoid_normalization"]))

            elif metric_name == "SO":
                metrics.append(Tooth.SurfaceOverlappingValues(num_classes=opt["classes"], sigmoid_normalization=opt["sigmoid_normalization"], theta=1.0))

            elif metric_name == "SD":
                metrics.append(Tooth.SurfaceDice(num_classes=opt["classes"], sigmoid_normalization=opt["sigmoid_normalization"], theta=1.0))

            elif metric_name == "IoU":
                metrics.append(Tooth.IoU(num_classes=opt["classes"], sigmoid_normalization=opt["sigmoid_normalization"]))

            else:
                raise RuntimeError(f"No {metric_name} metric available on {opt['dataset_name']} dataset")

    elif opt["dataset_name"] == "MMOTU":
        metrics = {}
        for metric_name in opt["metric_names"]:
            if metric_name == "DSC":
                metrics[metric_name] = MMOTU.DICE(num_classes=opt["classes"], sigmoid_normalization=opt["sigmoid_normalization"], mode=opt["dice_mode"])

            elif metric_name == "IoU":
                metrics[metric_name] = MMOTU.IoU(num_classes=opt["classes"], sigmoid_normalization=opt["sigmoid_normalization"])

            else:
                raise RuntimeError(f"No {metric_name} metric available on {opt['dataset_name']} dataset")

    elif opt["dataset_name"] == "ISIC-2018":
        metrics = {}
        for metric_name in opt["metric_names"]:
            if metric_name == "DSC":
                metrics[metric_name] = ISIC2018.DICE(num_classes=opt["classes"], sigmoid_normalization=opt["sigmoid_normalization"], mode=opt["dice_mode"])

            elif metric_name == "IoU":
                metrics[metric_name] = ISIC2018.IoU(num_classes=opt["classes"], sigmoid_normalization=opt["sigmoid_normalization"])

            elif metric_name == "JI":
                metrics[metric_name] = ISIC2018.JI(num_classes=opt["classes"], sigmoid_normalization=opt["sigmoid_normalization"])

            elif metric_name == "ACC":
                metrics[metric_name] = ISIC2018.ACC(num_classes=opt["classes"], sigmoid_normalization=opt["sigmoid_normalization"])

            else:
                raise RuntimeError(f"No {metric_name} metric available on {opt['dataset_name']} dataset")

    elif opt["dataset_name"] == "Tooth2D-X-Ray-6k":
        metrics = {}
        for metric_name in opt["metric_names"]:
            if metric_name == "DSC":
                metrics[metric_name] = ISIC2018.DICE(num_classes=opt["classes"], sigmoid_normalization=opt["sigmoid_normalization"], mode=opt["dice_mode"])

            elif metric_name == "IoU":
                metrics[metric_name] = ISIC2018.IoU(num_classes=opt["classes"], sigmoid_normalization=opt["sigmoid_normalization"])

            elif metric_name == "JI":
                metrics[metric_name] = ISIC2018.JI(num_classes=opt["classes"], sigmoid_normalization=opt["sigmoid_normalization"])

            elif metric_name == "ACC":
                metrics[metric_name] = ISIC2018.ACC(num_classes=opt["classes"], sigmoid_normalization=opt["sigmoid_normalization"])

            else:
                raise RuntimeError(f"No {metric_name} metric available on {opt['dataset_name']} dataset")

    else:
        raise RuntimeError(f"No {opt['dataset_name']} dataset available when initialize metrics")

    return metrics