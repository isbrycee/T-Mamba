# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   1378453948@qq.com
@DateTime :   2023/12/30 17:05
@Version  :   1.0
@License  :   (C)Copyright 2023
"""
import os
import argparse
import torch
from lib import utils, dataloaders, models, losses, metrics, trainers
import torch.distributed as dist
from datetime import datetime, timedelta

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

params_3D_CBCT_Tooth = {
    # ——————————————————————————————————————————————    Launch Initialization    —————————————————————————————————————————————————
    "CUDA_VISIBLE_DEVICES": "7",
    "seed": 1777777,
    "cuda": True,
    "benchmark": False, # False -> True
    "deterministic": True,
    # —————————————————————————————————————————————     Preprocessing      ————————————————————————————————————————————————————
    "resample_spacing": [0.5, 0.5, 0.5],
    "clip_lower_bound": -1412,
    "clip_upper_bound": 17943,
    "samples_train": 2048, # 2048
    "crop_size": (160, 160, 96),
    "crop_threshold": 0.5,
    # ——————————————————————————————————————————————    Data Augmentation    ——————————————————————————————————————————————————————
    "augmentation_probability": 0.3,
    "augmentation_method": "Choice",
    "open_elastic_transform": True,
    "elastic_transform_sigma": 20,
    "elastic_transform_alpha": 1,
    "open_gaussian_noise": True,
    "gaussian_noise_mean": 0,
    "gaussian_noise_std": 0.01,
    "open_random_flip": True,
    "open_random_rescale": True,
    "random_rescale_min_percentage": 0.5,
    "random_rescale_max_percentage": 1.5,
    "open_random_rotate": True,
    "random_rotate_min_angle": -50,
    "random_rotate_max_angle": 50,
    "open_random_shift": True,
    "random_shift_max_percentage": 0.3,
    "normalize_mean": 0.05029342141696459,
    "normalize_std": 0.028477091559295814,
    # —————————————————————————————————————————————    Data Loading     ——————————————————————————————————————————————————————
    "dataset_name": "3D-CBCT-Tooth",
    "dataset_path": r"./datasets/3D-CBCT-Tooth",
    "create_data": False,
    "batch_size": 6,
    "num_workers": 24,
    # —————————————————————————————————————————————    Model     ——————————————————————————————————————————————————————
    "model_name": "TMamba3D",
    "in_channels": 1,
    "classes": 2,
    "index_to_class_dict":
    {
        0: "background",
        1: "foreground"
    },
    "resume": None,
    # "resume": "/root/paddlejob/workspace/env_run/output/haojing08/PMFSNet-master-multigpu/runs/2024-03-26-16-09-15_DenseVNet_3D-CBCT-Tooth/checkpoints/latest_DenseVNet.state",
    "pretrain": None,
    # "pretrain": '/root/paddlejob/workspace/env_run/output/haojing08/PMFSNet-master-multigpu/runs/2024-03-26-16-09-15_DenseVNet_3D-CBCT-Tooth/checkpoints/latest_DenseVNet.pth',
    "high_frequency": 0.9,
    "low_frequency": 0.1,
    # ——————————————————————————————————————————————    Optimizer     ——————————————————————————————————————————————————————
    "optimizer_name": "AdamW",
    "learning_rate": 0.0005, # 0.0005
    "weight_decay": 0.00005,
    "momentum": 0.8,
    # ———————————————————————————————————————————    Learning Rate Scheduler     —————————————————————————————————————————————————————
    "lr_scheduler_name": "ReduceLROnPlateau",
    "gamma": 0.1,
    "step_size": 9,
    "milestones": [1, 3, 5, 7, 8, 9],
    "T_max": 2,
    "T_0": 2,
    "T_mult": 2,
    "mode": "max",
    "patience": 1,
    "factor": 0.5,
    # ————————————————————————————————————————————    Loss And Metric     ———————————————————————————————————————————————————————
    "metric_names": ["DSC"], # HD 
    "loss_function_name": "DiceLoss",
    "class_weight": [0.00551122, 0.99448878],
    "sigmoid_normalization": False,
    "dice_loss_mode": "extension",
    "dice_mode": "standard",
    # —————————————————————————————————————————————   Training   ——————————————————————————————————————————————————————
    "optimize_params": False,
    "use_amp": False,
    "run_dir": r"./runs",
    "start_epoch": 0,
    "end_epoch": 20,
    "best_dice": 0.60,
    "update_weight_freq": 1,
    "terminal_show_freq": 2,
    "save_epoch_freq": 4,
    # ————————————————————————————————————————————   Testing   ———————————————————————————————————————————————————————
    "crop_stride": [32, 32, 32]
}

params_MMOTU = {
    # ——————————————————————————————————————————————     Launch Initialization    ———————————————————————————————————————————————————
    "CUDA_VISIBLE_DEVICES": "0",
    "seed": 1777777,
    "cuda": True,
    "benchmark": False,
    "deterministic": True,
    # —————————————————————————————————————————————     Preprocessing       ————————————————————————————————————————————————————
    "resize_shape": (224, 224),
    # ——————————————————————————————————————————————    Data Augmentation    ——————————————————————————————————————————————————————
    "augmentation_p": 0.12097393901893663,
    "color_jitter": 0.4203933474361258,
    "random_rotation_angle": 30,
    "normalize_means": (0.22250386, 0.21844882, 0.21521868),
    "normalize_stds": (0.21923075, 0.21622984, 0.21370508),
    # —————————————————————————————————————————————    Data Loading     ——————————————————————————————————————————————————————
    "dataset_name": "MMOTU",
    "dataset_path": r"./datasets/MMOTU",
    "batch_size": 32,
    "num_workers": 2,
    # —————————————————————————————————————————————    Model     ——————————————————————————————————————————————————————
    "model_name": "PMFSNet",
    "in_channels": 3,
    "classes": 2,
    "index_to_class_dict":
    {
        0: "background",
        1: "foreground"
    },
    "resume": None,
    "pretrain": None,
    # ——————————————————————————————————————————————    Optimizer     ——————————————————————————————————————————————————————
    "optimizer_name": "AdamW",
    "learning_rate": 0.01,
    "weight_decay": 0.00001,
    "momentum": 0.7725414416309884,
    # ———————————————————————————————————————————    Learning Rate Scheduler     —————————————————————————————————————————————————————
    "lr_scheduler_name": "CosineAnnealingLR",
    "gamma": 0.8689275449032848,
    "step_size": 5,
    "milestones": [10, 30, 60, 100, 120, 140, 160, 170],
    "T_max": 200,
    "T_0": 10,
    "T_mult": 5,
    "mode": "max",
    "patience": 1,
    "factor": 0.97,
    # ————————————————————————————————————————————    Loss And Metric     ———————————————————————————————————————————————————————
    "metric_names": ["DSC", "IoU"],
    "loss_function_name": "DiceLoss",
    "class_weight": [0.2350689696563569, 1-0.2350689696563569],
    "sigmoid_normalization": False,
    "dice_loss_mode": "extension",
    "dice_mode": "standard",
    # —————————————————————————————————————————————   Training   ——————————————————————————————————————————————————————
    "optimize_params": False,
    "run_dir": r"./runs",
    "start_epoch": 0,
    "end_epoch": 2000,
    "best_metric": 0,
    "terminal_show_freq": 8,
    "save_epoch_freq": 500,
}

params_ISIC_2018 = {
    # ——————————————————————————————————————————————     Launch Initialization    ———————————————————————————————————————————————————
    "CUDA_VISIBLE_DEVICES": "0",
    "seed": 1777777,
    "cuda": True,
    "benchmark": False,
    "deterministic": True,
    # —————————————————————————————————————————————     Preprocessing       ————————————————————————————————————————————————————
    "resize_shape": (224, 224),
    # ——————————————————————————————————————————————    Data Augmentation    ——————————————————————————————————————————————————————
    "augmentation_p": 0.1,
    "color_jitter": 0.37,
    "random_rotation_angle": 15,
    "normalize_means": (0.50297405, 0.54711632, 0.71049083),
    "normalize_stds": (0.18653496, 0.17118206, 0.17080363),
    # —————————————————————————————————————————————    Data Loading     ——————————————————————————————————————————————————————
    "dataset_name": "ISIC-2018",
    "dataset_path": r"./datasets/ISIC-2018",
    "batch_size": 32,
    "num_workers": 2,
    # —————————————————————————————————————————————    Model     ——————————————————————————————————————————————————————
    "model_name": "PMFSNet",
    "in_channels": 3,
    "classes": 2,
    "index_to_class_dict":
    {
        0: "background",
        1: "foreground"
    },
    "resume": None,
    "pretrain": None,
    # ——————————————————————————————————————————————    Optimizer     ——————————————————————————————————————————————————————
    "optimizer_name": "AdamW",
    "learning_rate": 0.005,
    "weight_decay": 0.000001,
    "momentum": 0.9657205586290213,
    # ———————————————————————————————————————————    Learning Rate Scheduler     —————————————————————————————————————————————————————
    "lr_scheduler_name": "CosineAnnealingWarmRestarts",
    "gamma": 0.9582311026945434,
    "step_size": 20,
    "milestones": [1, 3, 5, 7, 8, 9],
    "T_max": 100,
    "T_0": 5,
    "T_mult": 5,
    "mode": "max",
    "patience": 20,
    "factor": 0.3,
    # ————————————————————————————————————————————    Loss And Metric     ———————————————————————————————————————————————————————
    "metric_names": ["DSC", "IoU", "JI", "ACC"],
    "loss_function_name": "DiceLoss",
    "class_weight": [0.029, 1-0.029],
    "sigmoid_normalization": False,
    "dice_loss_mode": "extension",
    "dice_mode": "standard",
    # —————————————————————————————————————————————   Training   ——————————————————————————————————————————————————————
    "optimize_params": False,
    "run_dir": r"./runs",
    "start_epoch": 0,
    "end_epoch": 150,
    "best_metric": 0,
    "terminal_show_freq": 20,
    "save_epoch_freq": 50,
}
params_Tooth_2D_X_ray = {
    # ——————————————————————————————————————————————     Launch Initialization    ———————————————————————————————————————————————————
    "CUDA_VISIBLE_DEVICES": "7",
    "seed": 1777777,
    "cuda": True,
    "benchmark": False,
    "deterministic": True,
    # —————————————————————————————————————————————     Preprocessing       ————————————————————————————————————————————————————
    "resize_shape": (640, 1280),
    # ——————————————————————————————————————————————    Data Augmentation    ——————————————————————————————————————————————————————
    "augmentation_p": 0.1,
    "color_jitter": 0.37,
    "random_rotation_angle": 15,
    "normalize_means": (0.50297405, 0.54711632, 0.71049083),
    "normalize_stds": (0.18653496, 0.17118206, 0.17080363),
    # —————————————————————————————————————————————    Data Loading     ——————————————————————————————————————————————————————
    "dataset_name": "Tooth2D-X-Ray-6k/",
    "dataset_path": r"./datasets/Tooth2D-X-Ray-6k/",
    "batch_size": 4,
    "num_workers": 4,
    # —————————————————————————————————————————————    Model     ——————————————————————————————————————————————————————
    "model_name": "TMamba2D",
    "in_channels": 3,
    "classes": 2,
    "index_to_class_dict":
    {
        0: "background",
        1: "foreground"
    },
    "resume": None,
    "pretrain": None,
    "high_frequency": 0.9,
    "low_frequency": 0.1,
    # ——————————————————————————————————————————————    Optimizer     ——————————————————————————————————————————————————————
    "optimizer_name": "AdamW",
    "learning_rate": 0.0075, # 0.005 0.0025
    # "weight_decay": 0.00005,
    # "momentum": 0.8,
    "weight_decay": 0.000001,
    "momentum": 0.9657205586290213,
    # ———————————————————————————————————————————    Learning Rate Scheduler     —————————————————————————————————————————————————————
    "lr_scheduler_name": "MultiStepLR",
    "gamma": 0.1,
    "step_size": 1,
    "milestones": [24, 28,], # [20, 26,]
    "T_max": 2,
    "T_0": 2,
    "T_mult": 2,
    "mode": "max",
    "patience": 1,
    "factor": 0.5,
    # ————————————————————————————————————————————    Loss And Metric     ———————————————————————————————————————————————————————
    "metric_names": ["DSC", "IoU", "JI", "ACC"],
    "loss_function_name": "DiceLoss",
    "class_weight": [0.029, 1-0.029],
    "sigmoid_normalization": False,
    "dice_loss_mode": "extension",
    "dice_mode": "standard",
    # —————————————————————————————————————————————   Training   ——————————————————————————————————————————————————————
    "optimize_params": False,
    "run_dir": r"./runs",
    "start_epoch": 0,
    "end_epoch": 30,
    "best_metric": 0,
    "update_weight_freq": 1,
    "terminal_show_freq": 1,
    "save_epoch_freq": 10,
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="3D-CBCT-Tooth", help="dataset name")
    parser.add_argument("--model", type=str, default="DenseVNet", help="model name")
    parser.add_argument("--pretrain_weight", type=str, default=None, help="pre-trained weight file path")
    parser.add_argument("--dimension", type=str, default="3d", help="dimension of dataset images and models")
    parser.add_argument("--scaling_version", type=str, default="TINY", help="scaling version of PMFSNet")
    parser.add_argument("--epoch", type=int, default=20, help="training epoch")
    parser.add_argument("--multi_gpu", type=str, default='false', help="using multi-gpu or not for training")
    parser.add_argument("--local-rank", type=int, default=0, help="local_rank")
    args = parser.parse_args()
    return args

def main():
    # analyse console arguments
    args = parse_args()
    if args.multi_gpu == 'True':
        torch.distributed.init_process_group(backend="nccl", timeout=timedelta(seconds=7200000))
        if args.local_rank == 0:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
    # select the dictionary of hyperparameters used for training
    if args.dataset == "3D-CBCT-Tooth":
        params = params_3D_CBCT_Tooth
    elif args.dataset == "MMOTU":
        params = params_MMOTU
    elif args.dataset == "ISIC-2018":
        params = params_ISIC_2018
    elif args.dataset == "Tooth2D-X-Ray-6k":
        params = params_Tooth_2D_X_ray
    else:
        raise RuntimeError(f"No {args.dataset} dataset available")

    # update the dictionary of hyperparameters used for training
    params["multi_gpu"] = args.multi_gpu == 'true' or args.multi_gpu == 'True'
    params["local_rank"] = args.local_rank
    params["dataset_name"] = args.dataset
    # for pretraining
    # params["dataset_path"] = os.path.join(r"/root/paddlejob/workspace/env_run/output/haojing08/PMFSNet-master-multigpu/datasets", ("CTooth+_labelled_CBCT" if args.dataset == "3D-CBCT-Tooth" else args.dataset))
    params["dataset_path"] = os.path.join(r"/root/paddlejob/workspace/env_run/output/haojing08/PMFSNet-master-multigpu/datasets", ("NC-release-data-checked" if args.dataset == "3D-CBCT-Tooth" else args.dataset))
    params["model_name"] = args.model
    if args.pretrain_weight is not None:
        params["pretrain"] = args.pretrain_weight
    params["dimension"] = args.dimension
    params["scaling_version"] = args.scaling_version

    if args.local_rank == 0:
        print('model type: ' + params["scaling_version"])
    if args.epoch is not None:
        params["end_epoch"] = args.epoch
        params["save_epoch_freq"] = args.epoch // 4

    # launch initialization
    os.environ["CUDA_VISIBLE_DEVICES"] = params["CUDA_VISIBLE_DEVICES"]
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    utils.reproducibility(params["seed"], params["deterministic"], params["benchmark"])
    
    # get the cuda device
    if params["cuda"]:
        if params["multi_gpu"]:
            params["device"] = torch.device("cuda:{}".format(params["local_rank"]) if torch.cuda.is_available() else "cpu")
        else:
            params["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        params["device"] = torch.device("cpu")
    curr_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    # print("Current Available GPU IDs: ", curr_cuda_visible_devices)
    if args.local_rank == 0:
        print("Complete the initialization of configuration")

    # initialize the dataloader
    train_loader, valid_loader = dataloaders.get_dataloader(params)
    if args.local_rank == 0:
        print("Complete the initialization of dataloader")

    # initialize the model, optimizer, and lr_scheduler
    model, optimizer, lr_scheduler = models.get_model_optimizer_lr_scheduler(params)
    if args.local_rank == 0:
        print("Complete the initialization of model:{}, optimizer:{}, and lr_scheduler:{}".format(params["model_name"], params["optimizer_name"], params["lr_scheduler_name"]))

    # initialize the loss function
    loss_function = losses.get_loss_function(params)
    if args.local_rank == 0:
        print("Complete the initialization of loss function")

    # initialize the metrics
    metric = metrics.get_metric(params)
    if args.local_rank == 0:
        print("Complete the initialization of metrics")

    # initialize the trainer
    trainer = trainers.get_trainer(params, train_loader, valid_loader, model, optimizer, lr_scheduler, loss_function, metric, )

    # resume or load pretrained weights
    if (params["resume"] is not None) or (params["pretrain"] is not None):
        trainer.load()
    if args.local_rank == 0:
        print("Complete the initialization of trainer")

    # start training
    trainer.training()


if __name__ == '__main__':
    main()


