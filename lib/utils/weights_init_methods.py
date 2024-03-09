# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   yykzhjh@163.com
@DateTime :   2022/11/14 15:25
@Version  :   1.0
@License  :   (C)Copyright 2022
"""
import torch.nn as nn


def weights_init_normal_3d(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None: nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Conv3d):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None: nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm, nn.InstanceNorm3d)):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


def weights_init_xavier_3d(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data, gain=1)
        if m.bias is not None: nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Conv3d):
        nn.init.xavier_normal_(m.weight.data, gain=1)
        if m.bias is not None: nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm, nn.InstanceNorm3d)):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


def weights_init_kaiming_3d(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        if m.bias is not None: nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Conv3d):
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        if m.bias is not None: nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm, nn.InstanceNorm3d)):
        if m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal_3d(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None: nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Conv3d):
        nn.init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None: nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm, nn.InstanceNorm3d)):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


def weights_init_normal_2d(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None: nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None: nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.InstanceNorm2d)):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


def weights_init_xavier_2d(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data, gain=1)
        if m.bias is not None: nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data, gain=1)
        if m.bias is not None: nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.InstanceNorm2d)):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


def weights_init_kaiming_2d(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        if m.bias is not None: nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        if m.bias is not None: nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.InstanceNorm2d)):
        if m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal_2d(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None: nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None: nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.InstanceNorm2d)):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


def init_weights(net, dim="3d", init_type='kaiming'):
    if init_type == 'normal':
        net.apply((weights_init_normal_3d if dim == "3d" else weights_init_normal_2d))
    elif init_type == 'xavier':
        net.apply((weights_init_xavier_3d if dim == "3d" else weights_init_xavier_2d))
    elif init_type == 'kaiming':
        net.apply((weights_init_kaiming_3d if dim == "3d" else weights_init_kaiming_2d))
    elif init_type == 'orthogonal':
        net.apply((weights_init_orthogonal_3d if dim == "3d" else weights_init_orthogonal_2d))
    else:
        raise NotImplementedError("No implementation of the [%s] initialization weight method" % init_type)
