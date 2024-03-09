import math
import numpy as np
import torch

import surface_distance as sd


def compute_per_channel_dice(input, target, mode="extension", epsilon=1e-6):
    """
    Computes DiceCoefficient as defined in https://arxiv.org/abs/1606.04797 given  a multi channel input and target.
    Assumes the input is a normalized probability, e.g. a result of Sigmoid or Softmax function.
    """
    assert input.size() == target.size(), "input and target are not the same shape"
    # (C, B * H * W * D)
    input = flatten(input)
    target = flatten(target)
    target = target.float()
    intersect = (input * target).sum(-1)
    # here we can use standard dice (input + target).sum(-1) or extension (see V-Net) (input^2 + target^2).sum(-1)
    if mode == "extension":
        denominator = (input * input).sum(-1) + (target * target).sum(-1)
    elif mode == "standard":
        denominator = input.sum(-1) + target.sum(-1)
    # print(denominator)
    return (2 * intersect + epsilon) / (denominator + epsilon)


def compute_per_channel_hd(seg, target, num_classes):
    bs, _, h, w, d = seg.shape
    output = torch.full((bs, num_classes), -1.0)

    for b in range(bs):
        for cla in range(num_classes):
            surface_distances = sd.compute_surface_distances(target[b, cla, ...].numpy(), seg[b, cla, ...].numpy(), spacing_mm=(1.0, 1.0, 1.0))

            if len(surface_distances["distances_pred_to_gt"]) == 0 or len(surface_distances["distances_gt_to_pred"]) == 0:
                continue
            output[b, cla] = sd.compute_robust_hausdorff(surface_distances, 95)
    out = torch.full((num_classes,), -1.0)
    for cla in range(num_classes):
        cnt = 0
        acc_sum = 0
        for b in range(bs):
            if output[b, cla] != -1.0:
                acc_sum += output[b, cla]
                cnt += 1.0
        out[cla] = acc_sum / cnt
    return out


def compute_per_channel_assd(seg, target, num_classes):
    bs, _, h, w, d = seg.shape
    output = torch.full((bs, num_classes), -1.0)

    for b in range(bs):
        for cla in range(num_classes):
            surface_distances = sd.compute_surface_distances(target[b, cla, ...].numpy(), seg[b, cla, ...].numpy(), spacing_mm=(1.0, 1.0, 1.0))

            if len(surface_distances["distances_pred_to_gt"]) == 0 or len(surface_distances["distances_gt_to_pred"]) == 0:
                continue
            assd_tuple = sd.compute_average_surface_distance(surface_distances)
            ASSD_per_class = ((assd_tuple[0] * len(surface_distances["distances_gt_to_pred"]) + assd_tuple[1] * len(surface_distances["distances_pred_to_gt"])) /
                              (len(surface_distances["distances_gt_to_pred"]) +
                               len(surface_distances["distances_pred_to_gt"])))
            output[b, cla] = ASSD_per_class
    out = torch.full((num_classes,), -1.0)
    for cla in range(num_classes):
        cnt = 0
        acc_sum = 0
        for b in range(bs):
            if output[b, cla] != -1.0:
                acc_sum += output[b, cla]
                cnt += 1.0
        out[cla] = acc_sum / cnt
    return out


def compute_per_channel_so(seg, target, num_classes, theta=1.0):
    bs, _, h, w, d = seg.shape
    output = torch.full((bs, num_classes), -1.0)

    for b in range(bs):
        for cla in range(num_classes):
            surface_distances = sd.compute_surface_distances(target[b, cla, ...].numpy(), seg[b, cla, ...].numpy(), spacing_mm=(1.0, 1.0, 1.0))

            if len(surface_distances["distances_pred_to_gt"]) == 0 or len(surface_distances["distances_gt_to_pred"]) == 0:
                continue
            so_tuple = sd.compute_surface_overlap_at_tolerance(surface_distances, tolerance_mm=theta)
            output[b, cla] = so_tuple[1]
    out = torch.full((num_classes,), -1.0)
    for cla in range(num_classes):
        cnt = 0
        acc_sum = 0
        for b in range(bs):
            if output[b, cla] != -1.0:
                acc_sum += output[b, cla]
                cnt += 1.0
        out[cla] = acc_sum / cnt
    return out


def compute_per_channel_sd(seg, target, num_classes, theta=1.0):
    bs, _, h, w, d = seg.shape
    output = torch.full((bs, num_classes), -1.0)

    for b in range(bs):
        for cla in range(num_classes):
            surface_distances = sd.compute_surface_distances(target[b, cla, ...].numpy(), seg[b, cla, ...].numpy(), spacing_mm=(1.0, 1.0, 1.0))

            if len(surface_distances["distances_pred_to_gt"]) == 0 or len(surface_distances["distances_gt_to_pred"]) == 0:
                continue
            sd_score = sd.compute_surface_dice_at_tolerance(surface_distances, tolerance_mm=theta)
            output[b, cla] = sd_score
    out = torch.full((num_classes,), -1.0)
    for cla in range(num_classes):
        cnt = 0
        acc_sum = 0
        for b in range(bs):
            if output[b, cla] != -1.0:
                acc_sum += output[b, cla]
                cnt += 1.0
        out[cla] = acc_sum / cnt
    return out


def compute_per_channel_iou(seg, target, num_classes):
    bs, _, h, w, d = seg.shape
    output = torch.full((bs, num_classes), -1.0)

    for b in range(bs):
        for cla in range(num_classes):
            intersection = np.logical_and(target[b, cla, ...].numpy(), seg[b, cla, ...].numpy())
            union = np.logical_or(target[b, cla, ...].numpy(), seg[b, cla, ...].numpy())
            iou_score = np.sum(intersection) / np.sum(union)
            output[b, cla] = iou_score
    out = torch.full((num_classes,), -1.0)
    for cla in range(num_classes):
        cnt = 0
        acc_sum = 0
        for b in range(bs):
            if output[b, cla] != -1.0:
                acc_sum += output[b, cla]
                cnt += 1.0
        out[cla] = acc_sum / cnt
    return out


def cal_dsc(result, reference):
    r"""
    Dice coefficient

    Computes the Dice coefficient (also known as Sorensen index) between the binary
    objects in two images.

    The metric is defined as

    .. math::

        DC=\frac{2|A\cap B|}{|A|+|B|}

    , where :math:`A` is the first and :math:`B` the second set of samples (here: binary objects).

    Parameters
    ----------
    result : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    reference : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.

    Returns
    -------
    dc : float
        The Dice coefficient between the object(s) in ```result``` and the
        object(s) in ```reference```. It ranges from 0 (no overlap) to 1 (perfect overlap).

    Notes
    -----
    This is a real metric. The binary images can therefore be supplied in any order.
    """
    result = np.atleast_1d(result.astype(bool))
    reference = np.atleast_1d(reference.astype(bool))

    intersection = np.count_nonzero(result & reference)

    size_i1 = np.count_nonzero(result)
    size_i2 = np.count_nonzero(reference)

    try:
        dc = 2. * intersection / (float(size_i1 + size_i2) + 1e-6)
    except ZeroDivisionError:
        dc = 0.0

    return dc


def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, H, W, D) -> (C, N * H * W * D)
    """
    # number of channels
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)
