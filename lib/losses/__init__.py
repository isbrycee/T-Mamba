# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   1378453948@qq.com
@DateTime :   2023/12/30 16:56
@Version  :   1.0
@License  :   (C)Copyright 2023
"""
import torch

from .DiceLoss import DiceLoss


def get_loss_function(opt):
    if opt["loss_function_name"] == "DiceLoss":
        loss_function = DiceLoss(opt["classes"], weight=torch.FloatTensor(opt["class_weight"]).to(opt["device"]),
                                 sigmoid_normalization=False, mode=opt["dice_loss_mode"])

    else:
        raise RuntimeError(f"No {opt['loss_function_name']} is available")

    return loss_function
