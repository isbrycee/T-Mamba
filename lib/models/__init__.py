# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   1378453948@qq.com
@DateTime :   2023/12/30 16:57
@Version  :   1.0
@License  :   (C)Copyright 2023
"""
import torch
import torch.optim as optim
import torch.nn as nn

import lib.utils as utils
import lib.models.nnunetv2 as nnunetv2

from .DenseVNet import DenseVNet
from .DenseVNet2D import DenseVNet2D
from .TMamba3D import TMamba3D
from .TMamba2D import TMamba2D
from .UNet3D import UNet3D
from .VNet import VNet
from .AttentionUNet3D import AttentionUNet3D
from .R2UNet import R2U_Net
from .R2AttentionUNet import R2AttentionU_Net
from .HighResNet3D import HighResNet3D
from .DenseVoxelNet import DenseVoxelNet
from .MultiResUNet3D import MultiResUNet3D
from .DenseASPPUNet import DenseASPPUNet
from .TransBTS import BTS
from monai.networks.nets import UNETR, SwinUNETR
from lib.models.nnFormer.nnFormer_seg import nnFormer
from lib.models.UXNet_3D.network_backbone import UXNET
from monai.networks.nets import AttentionUnet

from .PSPNet import PSPNet
from .DANet import DANet
from .SegFormer import SegFormer
from .UNet import UNet
from .TransUNet import TransUNet
from .TransUNet import CONFIGS as CONFIGS_ViT_seg
from .BiSeNetV2 import BiSeNetV2
from .MedT import MedT

from .AttU_Net import AttU_Net
from .CANet import Comprehensive_Atten_Unet
from .BCDUNet import BCDUNet
from .CENet import CE_Net
from .CPFNet import CPF_Net
from .CKDNet import DeepLab_Aux
from .MsRED import Ms_red_v1, Ms_red_v2
from .MobileNetV2 import MobileNetV2

from .PMFSNet import PMFSNet
from .SwinUMamba import SwinUMamba

def get_model_optimizer_lr_scheduler(opt):
    # initialize model
    if opt["dataset_name"] == "3D-CBCT-Tooth":
        if opt["model_name"] == "DenseVNet":
            model = DenseVNet(in_channels=opt["in_channels"], classes=opt["classes"])
        if opt["model_name"] == "TMamba3D":
            model = TMamba3D(in_channels=opt["in_channels"], classes=opt["classes"], input_size=opt['crop_size'], high_freq=opt['high_frequency'], low_freq=opt['low_frequency'])

        elif opt["model_name"] == "UNet3D":
            model = UNet3D(opt["in_channels"], opt["classes"], final_sigmoid=False)

        elif opt["model_name"] == "VNet":
            model = VNet(in_channels=opt["in_channels"], classes=opt["classes"])

        elif opt["model_name"] == "AttentionUNet3D":
            # model = AttentionUNet3D(in_channels=opt["in_channels"], out_channels=opt["classes"])
            model = AttentionUnet(spatial_dims=3, in_channels=opt["in_channels"], out_channels=opt["classes"], channels=(64, 128, 256, 512, 1024), strides=(2, 2, 2, 2))

        elif opt["model_name"] == "R2UNet":
            model = R2U_Net(in_channels=opt["in_channels"], out_channels=opt["classes"])

        elif opt["model_name"] == "R2AttentionUNet":
            model = R2AttentionU_Net(in_channels=opt["in_channels"], out_channels=opt["classes"])

        elif opt["model_name"] == "HighResNet3D":
            model = HighResNet3D(in_channels=opt["in_channels"], classes=opt["classes"])

        elif opt["model_name"] == "DenseVoxelNet":
            model = DenseVoxelNet(in_channels=opt["in_channels"], classes=opt["classes"])

        elif opt["model_name"] == "MultiResUNet3D":
            model = MultiResUNet3D(in_channels=opt["in_channels"], classes=opt["classes"])

        elif opt["model_name"] == "DenseASPPUNet":
            model = DenseASPPUNet(in_channels=opt["in_channels"], classes=opt["classes"])

        elif opt["model_name"] == "UNETR":
            model = UNETR(
                in_channels=opt["in_channels"],
                out_channels=opt["classes"],
                img_size=(160, 160, 96),
                feature_size=16,
                hidden_size=768,
                mlp_dim=3072,
                num_heads=12,
                pos_embed="perceptron",
                norm_name="instance",
                res_block=True,
                dropout_rate=0.0,
            )

        elif opt["model_name"] == "SwinUNETR":
            model = SwinUNETR(
                img_size=(160, 160, 96),
                in_channels=opt["in_channels"],
                out_channels=opt["classes"],
                feature_size=48,
                use_checkpoint=False,
            )

        elif opt["model_name"] == "TransBTS":
            model = BTS(img_dim=(160, 160, 96), patch_dim=8, num_channels=opt["in_channels"], num_classes=opt["classes"],
                        embedding_dim=512,
                        num_heads=8,
                        num_layers=4,
                        hidden_dim=4096,
                        dropout_rate=0.1,
                        attn_dropout_rate=0.1,
                        conv_patch_representation=True,
                        positional_encoding_type="learned",
                        )

        elif opt["model_name"] == "nnFormer":
            model = nnFormer(crop_size=(160, 160, 96), input_channels=opt["in_channels"], num_classes=opt["classes"])

        elif opt["model_name"] == "3DUXNet":
            model = UXNET(
                in_chans=opt["in_channels"],
                out_chans=opt["classes"],
                depths=[2, 2, 2, 2],
                feat_size=[48, 96, 192, 384],
                drop_path_rate=0,
                layer_scale_init_value=1e-6,
                spatial_dims=3,
            )

        elif opt["model_name"] == "PMFSNet":
            # model = PMFSNet(in_channels=opt["in_channels"], out_channels=opt["classes"], dim=opt["dimension"], scaling_version=opt["scaling_version"])
            raise NotImplementedError("PMFSNet has not yet been implemented")

        else:
            raise RuntimeError(f"No {opt['model_name']} model available on {opt['dataset_name']} dataset")

    elif opt["dataset_name"] == "Tooth2D-X-Ray-6k":
        if opt["model_name"] == "TMamba2D":
            model = TMamba2D(in_channels=opt["in_channels"], classes=opt["classes"], scaling_version=opt["scaling_version"], input_size=opt['resize_shape'], high_freq=opt['high_frequency'], low_freq=opt['low_frequency'])
        elif opt["model_name"] == "UNet":
            model = UNet(n_channels=opt["in_channels"], n_classes=opt["classes"])
        elif opt["model_name"] == "CKDNet":
            model = DeepLab_Aux(num_classes=opt["classes"])
        elif opt["model_name"] == "AttU_Net":
            model = AttU_Net(img_ch=opt["in_channels"], output_ch=opt["classes"])
        elif opt["model_name"] == "BCDUNet":
            model = BCDUNet(output_dim=opt["classes"], input_dim=opt["in_channels"], frame_size=opt["resize_shape"])
        elif opt["model_name"] == "CPFNet":
            model = CPF_Net(classes=opt["classes"], channels=opt["in_channels"])
        elif opt["model_name"] == "CENet":
            model = CE_Net(classes=opt["classes"], channels=opt["in_channels"])
        elif opt["model_name"] == "MsRED":
            model = Ms_red_v2(classes=opt["classes"], channels=opt["in_channels"], out_size=opt["resize_shape"])
        elif opt["model_name"] == "SegFormer":
            model = SegFormer(channels=opt["in_channels"], num_classes=opt["classes"])
        elif opt["model_name"] == "BiSeNetV2":
            model = BiSeNetV2(n_classes=opt["classes"])
        elif opt["model_name"] == "PMFSNet":
            model = PMFSNet(in_channels=opt["in_channels"], out_channels=opt["classes"], dim=opt["dimension"], scaling_version=opt["scaling_version"])
        elif opt["model_name"] == "SwinUMamba":
            model = SwinUMamba(in_chans=opt["in_channels"], out_chans=opt["classes"])
        else:
            raise RuntimeError(f"No {opt['model_name']} model available on {opt['dataset_name']} dataset")

    elif opt["dataset_name"] == "MMOTU":
        if opt["model_name"] == "PMFSNet":
            # model = PMFSNet(in_channels=opt["in_channels"], out_channels=opt["classes"], dim=opt["dimension"], scaling_version=opt["scaling_version"])
            raise NotImplementedError("PMFcdSNet has not yet been implemented")

        elif opt["model_name"] == "MobileNetV2":
            model = MobileNetV2(in_channels=opt["in_channels"], out_channels=opt["classes"], input_size=opt["resize_shape"][0], width_mult=1.)

        elif opt["model_name"] == "PSPNet":
            model = PSPNet(n_classes=opt["classes"], backend='resnet50', pretrained=True)

        elif opt["model_name"] == "DANet":
            model = DANet(nclass=opt["classes"])

        elif opt["model_name"] == "SegFormer":
            model = SegFormer(channels=opt["in_channels"], num_classes=opt["classes"])

        elif opt["model_name"] == "UNet":
            model = UNet(n_channels=opt["in_channels"], n_classes=opt["classes"])

        elif opt["model_name"] == "TransUNet":
            config_vit = CONFIGS_ViT_seg["R50-ViT-B_16"]
            config_vit.n_classes = opt["classes"]
            config_vit.n_skip = 3
            config_vit.patches.grid = (int(opt["resize_shape"][0] / 16), int(opt["resize_shape"][1] / 16))
            model = TransUNet(config_vit, img_size=opt["resize_shape"][0], num_classes=config_vit.n_classes)

        elif opt["model_name"] == "BiSeNetV2":
            model = BiSeNetV2(n_classes=opt["classes"])

        elif opt["model_name"] == "MedT":
            model = MedT(imgchan=opt["in_channels"], num_classes=opt["classes"])

        else:
            raise RuntimeError(f"No {opt['model_name']} model available on {opt['dataset_name']} dataset")

    elif opt["dataset_name"] == "ISIC-2018":
        if opt["model_name"] == "PMFSNet":
            # model = PMFSNet(in_channels=opt["in_channels"], out_channels=opt["classes"], dim=opt["dimension"], scaling_version=opt["scaling_version"])
            raise NotImplementedError("PMFSNet has not yet been implemented")

        elif opt["model_name"] == "MobileNetV2":
            model = MobileNetV2(in_channels=opt["in_channels"], out_channels=opt["classes"], input_size=opt["resize_shape"][0], width_mult=1.)

        elif opt["model_name"] == "UNet":
            model = UNet(n_channels=opt["in_channels"], n_classes=opt["classes"])

        elif opt["model_name"] == "MsRED":
            model = Ms_red_v2(classes=opt["classes"], channels=opt["in_channels"], out_size=opt["resize_shape"])

        elif opt["model_name"] == "CKDNet":
            model = DeepLab_Aux(num_classes=opt["classes"])

        elif opt["model_name"] == "BCDUNet":
            model = BCDUNet(output_dim=opt["classes"], input_dim=opt["in_channels"], frame_size=opt["resize_shape"])

        elif opt["model_name"] == "CANet":
            model = Comprehensive_Atten_Unet(in_ch=opt["in_channels"], n_classes=opt["classes"], out_size=opt["resize_shape"])

        elif opt["model_name"] == "CENet":
            model = CE_Net(classes=opt["classes"], channels=opt["in_channels"])

        elif opt["model_name"] == "CPFNet":
            model = CPF_Net(classes=opt["classes"], channels=opt["in_channels"])

        elif opt["model_name"] == "AttU_Net":
            model = AttU_Net(img_ch=opt["in_channels"], output_ch=opt["classes"])
        else:
            raise RuntimeError(f"No {opt['model_name']} model available on {opt['dataset_name']} dataset")

    else:
        raise RuntimeError(f"No {opt['dataset_name']} dataset available when initialize model")

    # initialize model and weights
    if not opt['multi_gpu']:
        model = model.to(opt["device"])
        utils.init_weights(model, init_type="kaiming")
    else:
        # DDP
        from torch.utils.data.distributed import DistributedSampler
        # torch.distributed.init_process_group(backend="nccl")
        local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = model.to(device)
        utils.init_weights(model, init_type="kaiming") # xavier kaiming
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[local_rank],
                                                      output_device=local_rank,
                                                      find_unused_parameters=True)

        # model = model.to(device)
        # utils.init_weights(model, init_type="kaiming") # xavier kaiming
        # num_gpus = torch.cuda.device_count()
        # available_gpu_ids = list(range(num_gpus))
        # model = torch.nn.DataParallel(model, device_ids=available_gpu_ids)

    # initialize optimizer
    if opt["optimizer_name"] == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=opt["learning_rate"], momentum=opt["momentum"],
                              weight_decay=opt["weight_decay"])

    elif opt["optimizer_name"] == 'Adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=opt["learning_rate"], weight_decay=opt["weight_decay"])

    elif opt["optimizer_name"] == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=opt["learning_rate"], weight_decay=opt["weight_decay"],
                                  momentum=opt["momentum"])

    elif opt["optimizer_name"] == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=opt["learning_rate"], weight_decay=opt["weight_decay"])

    elif opt["optimizer_name"] == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=opt["learning_rate"], weight_decay=opt["weight_decay"])

    elif opt["optimizer_name"] == "Adamax":
        optimizer = optim.Adamax(model.parameters(), lr=opt["learning_rate"], weight_decay=opt["weight_decay"])

    elif opt["optimizer_name"] == "Adadelta":
        optimizer = optim.Adadelta(model.parameters(), lr=opt["learning_rate"], weight_decay=opt["weight_decay"])

    else:
        raise RuntimeError(f"No {opt['optimizer_name']} optimizer available")

    # initialize lr_scheduler
    if opt["lr_scheduler_name"] == "ExponentialLR":
        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=opt["gamma"])

    elif opt["lr_scheduler_name"] == "StepLR":
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt["step_size"], gamma=opt["gamma"])

    elif opt["lr_scheduler_name"] == "MultiStepLR":
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt["milestones"], gamma=opt["gamma"])

    elif opt["lr_scheduler_name"] == "CosineAnnealingLR":
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt["T_max"])

    elif opt["lr_scheduler_name"] == "CosineAnnealingWarmRestarts":
        lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=opt["T_0"],
                                                                      T_mult=opt["T_mult"])

    elif opt["lr_scheduler_name"] == "OneCycleLR":
        lr_scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=opt["learning_rate"],
                                                     steps_per_epoch=opt["steps_per_epoch"], epochs=opt["end_epoch"], cycle_momentum=False)

    elif opt["lr_scheduler_name"] == "ReduceLROnPlateau":
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=opt["mode"], factor=opt["factor"],
                                                            patience=opt["patience"])
    else:
        raise RuntimeError(f"No {opt['lr_scheduler_name']} lr_scheduler available")

    return model, optimizer, lr_scheduler


def get_model(opt):
    # initialize model
    if opt["dataset_name"] == "3D-CBCT-Tooth":
        if opt["model_name"] == "DenseVNet":
            model = DenseVNet(in_channels=opt["in_channels"], classes=opt["classes"])
        elif opt["model_name"] == "TMamba3D":
            model = TMamba3D(in_channels=opt["in_channels"], classes=opt["classes"], input_size=opt['crop_size'], high_freq=opt['high_frequency'], low_freq=opt['low_frequency'])
        elif opt["model_name"] == "UNet3D":
            model = UNet3D(opt["in_channels"], opt["classes"], final_sigmoid=False)

        elif opt["model_name"] == "VNet":
            model = VNet(in_channels=opt["in_channels"], classes=opt["classes"])

        elif opt["model_name"] == "AttentionUNet3D":
            # model = AttentionUNet3D(in_channels=opt["in_channels"], out_channels=opt["classes"])
            model = AttentionUnet(spatial_dims=3, in_channels=opt["in_channels"], out_channels=opt["classes"], channels=(64, 128, 256, 512, 1024), strides=(2, 2, 2, 2))

        elif opt["model_name"] == "R2UNet":
            model = R2U_Net(in_channels=opt["in_channels"], out_channels=opt["classes"])

        elif opt["model_name"] == "R2AttentionUNet":
            model = R2AttentionU_Net(in_channels=opt["in_channels"], out_channels=opt["classes"])

        elif opt["model_name"] == "HighResNet3D":
            model = HighResNet3D(in_channels=opt["in_channels"], classes=opt["classes"])

        elif opt["model_name"] == "DenseVoxelNet":
            model = DenseVoxelNet(in_channels=opt["in_channels"], classes=opt["classes"])

        elif opt["model_name"] == "MultiResUNet3D":
            model = MultiResUNet3D(in_channels=opt["in_channels"], classes=opt["classes"])

        elif opt["model_name"] == "DenseASPPUNet":
            model = DenseASPPUNet(in_channels=opt["in_channels"], classes=opt["classes"])

        elif opt["model_name"] == "UNETR":
            model = UNETR(
                in_channels=opt["in_channels"],
                out_channels=opt["classes"],
                img_size=(160, 160, 96),
                feature_size=16,
                hidden_size=768,
                mlp_dim=3072,
                num_heads=12,
                pos_embed="perceptron",
                norm_name="instance",
                res_block=True,
                dropout_rate=0.0,
            )

        elif opt["model_name"] == "SwinUNETR":
            model = SwinUNETR(
                img_size=(160, 160, 96),
                in_channels=opt["in_channels"],
                out_channels=opt["classes"],
                feature_size=48,
                use_checkpoint=False,
            )

        elif opt["model_name"] == "TransBTS":
            model = BTS(img_dim=(160, 160, 96), patch_dim=8, num_channels=opt["in_channels"], num_classes=opt["classes"],
                        embedding_dim=512,
                        num_heads=8,
                        num_layers=4,
                        hidden_dim=4096,
                        dropout_rate=0.1,
                        attn_dropout_rate=0.1,
                        conv_patch_representation=True,
                        positional_encoding_type="learned",
                        )

        elif opt["model_name"] == "nnFormer":
            model = nnFormer(crop_size=(160, 160, 96), input_channels=opt["in_channels"], num_classes=opt["classes"])

        elif opt["model_name"] == "3DUXNet":
            model = UXNET(
                in_chans=opt["in_channels"],
                out_chans=opt["classes"],
                depths=[2, 2, 2, 2],
                feat_size=[48, 96, 192, 384],
                drop_path_rate=0,
                layer_scale_init_value=1e-6,
                spatial_dims=3,
            )

        elif opt["model_name"] == "PMFSNet":
            # model = PMFSNet(in_channels=opt["in_channels"], out_channels=opt["classes"], dim=opt["dimension"], scaling_version=opt["scaling_version"])
            raise NotImplementedError("PMFSNet has not yet been implemented")

        else:
            raise RuntimeError(f"No {opt['model_name']} model available on {opt['dataset_name']} dataset")

    elif opt["dataset_name"] == "MMOTU":
        if opt["model_name"] == "PMFSNet":
            # model = PMFSNet(in_channels=opt["in_channels"], out_channels=opt["classes"], dim=opt["dimension"], scaling_version=opt["scaling_version"])
            raise NotImplementedError("PMFSNet has not yet been implemented")

        elif opt["model_name"] == "MobileNetV2":
            model = MobileNetV2(in_channels=opt["in_channels"], out_channels=opt["classes"], input_size=opt["resize_shape"][0], width_mult=1.)

        elif opt["model_name"] == "PSPNet":
            model = PSPNet(n_classes=opt["classes"], backend='resnet50', pretrained=True)

        elif opt["model_name"] == "DANet":
            model = DANet(nclass=opt["classes"])

        elif opt["model_name"] == "SegFormer":
            model = SegFormer(channels=opt["in_channels"], num_classes=opt["classes"])

        elif opt["model_name"] == "UNet":
            model = UNet(n_channels=opt["in_channels"], n_classes=opt["classes"])

        elif opt["model_name"] == "TransUNet":
            config_vit = CONFIGS_ViT_seg["R50-ViT-B_16"]
            config_vit.n_classes = opt["classes"]
            config_vit.n_skip = 3
            config_vit.patches.grid = (int(opt["resize_shape"][0] / 16), int(opt["resize_shape"][1] / 16))
            model = TransUNet(config_vit, img_size=opt["resize_shape"][0], num_classes=config_vit.n_classes)

        elif opt["model_name"] == "BiSeNetV2":
            model = BiSeNetV2(n_classes=opt["classes"])

        elif opt["model_name"] == "MedT":
            model = MedT(imgchan=opt["in_channels"], num_classes=opt["classes"])

        else:
            raise RuntimeError(f"No {opt['model_name']} model available on {opt['dataset_name']} dataset")

    elif (opt["dataset_name"] == "ISIC-2018") or (opt["dataset_name"] == "Tooth2D-X-Ray-6k"):
        if opt["model_name"] == "PMFSNet":
            # model = PMFSNet(in_channels=opt["in_channels"], out_channels=opt["classes"], dim=opt["dimension"], scaling_version=opt["scaling_version"])
            raise NotImplementedError("PMFSNet has not yet been implemented")

        elif opt["model_name"] == "MobileNetV2":
            model = MobileNetV2(in_channels=opt["in_channels"], out_channels=opt["classes"], input_size=opt["resize_shape"][0], width_mult=1.)

        elif opt["model_name"] == "UNet":
            model = UNet(n_channels=opt["in_channels"], n_classes=opt["classes"])

        elif opt["model_name"] == "MsRED":
            model = Ms_red_v2(classes=opt["classes"], channels=opt["in_channels"], out_size=opt["resize_shape"])

        elif opt["model_name"] == "CKDNet":
            model = DeepLab_Aux(num_classes=opt["classes"])

        elif opt["model_name"] == "BCDUNet":
            model = BCDUNet(output_dim=opt["classes"], input_dim=opt["in_channels"], frame_size=opt["resize_shape"])

        elif opt["model_name"] == "CANet":
            model = Comprehensive_Atten_Unet(in_ch=opt["in_channels"], n_classes=opt["classes"], out_size=opt["resize_shape"])

        elif opt["model_name"] == "CENet":
            model = CE_Net(classes=opt["classes"], channels=opt["in_channels"])

        elif opt["model_name"] == "CPFNet":
            model = CPF_Net(classes=opt["classes"], channels=opt["in_channels"])

        elif opt["model_name"] == "AttU_Net":
            model = AttU_Net(img_ch=opt["in_channels"], output_ch=opt["classes"])

        elif opt["model_name"] == "BiSeNetV2":
            model = BiSeNetV2(n_classes=opt["classes"])
            
        elif opt["model_name"] == "TMamba2D":
            model = TMamba2D(in_channels=opt["in_channels"], classes=opt["classes"], scaling_version=opt["scaling_version"], input_size=opt['resize_shape'], high_freq=opt['high_frequency'], low_freq=opt['low_frequency'])
        else:
            raise RuntimeError(f"No {opt['model_name']} model available on {opt['dataset_name']} dataset")

    else:
        raise RuntimeError(f"No {opt['dataset_name']} dataset available when initialize model")

    model = model.to(opt["device"])

    return model
