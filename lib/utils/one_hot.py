# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   yykzhjh@163.com
@DateTime :   2023/5/1 18:26
@Version  :   1.0
@License  :   (C)Copyright 2023
"""
import torch


def expand_as_one_hot(input, C):
    input = input.unsqueeze(1)
    shape = list(input.size())
    shape[1] = C

    return torch.zeros(shape).to(input.device).scatter_(1, input, 1)
