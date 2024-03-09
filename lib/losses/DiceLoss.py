from lib.utils import *
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """Computes Dice Loss according to https://arxiv.org/abs/1606.04797.
    For multi-class segmentation `weight` parameter can be used to assign different weights per class.
    """

    def __init__(self, classes=1, weight=None, sigmoid_normalization=False, mode="extension"):
        super(DiceLoss, self).__init__()
        self.classes = classes
        self.weight = weight
        self.mode = mode
        
        if sigmoid_normalization:
            self.normalization = nn.Sigmoid()
        else:
            self.normalization = nn.Softmax(dim=1)
        
    def dice(self, input, target):
        target = expand_as_one_hot(target.long(), self.classes)

        # foe segformer
        if input.size()[3] == (target.size()[3] / 4):
            input = F.interpolate(input, size=(640, 1280), mode='bilinear', align_corners=False)
        assert input.size() == target.size(), "Inconsistency of dimensions between predicted and labeled images after one-hot processing in dice loss"
        input = self.normalization(input)
        return compute_per_channel_dice(input, target, epsilon=1e-6, mode=self.mode)


    def forward(self, input, target):
        per_channel_dice = self.dice(input, target)
        real_weight = self.weight.clone()
        for i, dice in enumerate(per_channel_dice):
            if dice == 0:
                real_weight[i] = 0

        weighted_dsc = torch.sum(per_channel_dice * real_weight) / torch.sum(real_weight)

        loss = 1. - weighted_dsc

        return loss
