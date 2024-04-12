# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   1378453948@qq.com
@DateTime :   2023/12/30 16:56
@Version  :   1.0
@License  :   (C)Copyright 2023
"""
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from .ToothDataset import ToothDataset
from .MMOTUDataset import MMOTUDataset
from .ISIC2018Dataset import ISIC2018Dataset


def get_dataloader(opt):
    """
    get dataloader
    Args:
        opt: params dict
    Returns:
    """
    if opt["dataset_name"] == "3D-CBCT-Tooth":
        train_set = ToothDataset(opt, mode="train")
        valid_set = ToothDataset(opt, mode="valid")

        train_loader = DataLoader(train_set, batch_size=opt["batch_size"], shuffle=True, num_workers=opt["num_workers"], pin_memory=True)
        valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False, num_workers=opt["num_workers"], pin_memory=True)
        # valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    elif opt["dataset_name"] == "MMOTU":
        train_set = MMOTUDataset(opt, mode="train")
        valid_set = MMOTUDataset(opt, mode="valid")

        train_loader = DataLoader(train_set, batch_size=opt["batch_size"], shuffle=True, num_workers=opt["num_workers"], pin_memory=True)
        valid_loader = DataLoader(valid_set, batch_size=opt["batch_size"], shuffle=False, num_workers=opt["num_workers"], pin_memory=True)

    elif opt["dataset_name"] == "ISIC-2018":
        train_set = ISIC2018Dataset(opt, mode="train")
        valid_set = ISIC2018Dataset(opt, mode="valid")

        train_loader = DataLoader(train_set, batch_size=opt["batch_size"], shuffle=True, num_workers=opt["num_workers"], pin_memory=True)
        valid_loader = DataLoader(valid_set, batch_size=opt["batch_size"], shuffle=False, num_workers=opt["num_workers"], pin_memory=True)

    elif opt["dataset_name"] == "Tooth2D-X-Ray-6k":
        train_set = ISIC2018Dataset(opt, mode="train")
        valid_set = ISIC2018Dataset(opt, mode="valid")
        if not opt['multi_gpu']:
            train_loader = DataLoader(train_set, batch_size=opt["batch_size"], shuffle=True, num_workers=opt["num_workers"], pin_memory=True)
            valid_loader = DataLoader(valid_set, batch_size=opt["batch_size"], shuffle=False, num_workers=opt["num_workers"], pin_memory=True)
        else:
            train_loader = DataLoader(dataset=train_set,
                            batch_size=opt["batch_size"],
                            sampler=DistributedSampler(train_set),
                            num_workers=opt["num_workers"]
                            )
            valid_loader = DataLoader(valid_set, batch_size=opt["batch_size"], shuffle=False, num_workers=opt["num_workers"], pin_memory=True)
            # valid_loader = DataLoader(dataset=valid_set,
            #                 batch_size=opt["batch_size"],
            #                 sampler=DistributedSampler(valid_set),
            #                 num_workers=opt["num_workers"])
    else:
        raise RuntimeError(f"No {opt['dataset_name']} dataloader available")

    opt["steps_per_epoch"] = len(train_loader)

    return train_loader, valid_loader


def get_test_dataloader(opt):
    """
    get test dataloader
    :param opt: params dict
    :return:
    """
    if opt["dataset_name"] == "3D-CBCT-Tooth":
        valid_set = ToothDataset(opt, mode="valid")
        valid_loader = DataLoader(valid_set, batch_size=opt["batch_size"], shuffle=False, num_workers=1, pin_memory=True)

    elif opt["dataset_name"] == "MMOTU":
        valid_set = MMOTUDataset(opt, mode="valid")
        valid_loader = DataLoader(valid_set, batch_size=opt["batch_size"], shuffle=False, num_workers=1, pin_memory=True)

    elif opt["dataset_name"] == "ISIC-2018":
        valid_set = ISIC2018Dataset(opt, mode="valid")
        valid_loader = DataLoader(valid_set, batch_size=opt["batch_size"], shuffle=False, num_workers=1, pin_memory=True)

    elif opt["dataset_name"] == "Tooth2D-X-Ray-6k":
        valid_set = ISIC2018Dataset(opt, mode="valid")
        valid_loader = DataLoader(valid_set, batch_size=opt["batch_size"], shuffle=False, num_workers=opt["num_workers"], pin_memory=True)

    else:
        raise RuntimeError(f"No {opt['dataset_name']} dataloader available")

    return valid_loader